import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor

class TravelModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model = load_traj_model(model_args)
        self.model.to(torch.bfloat16)
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    def _tokenize_instruction_units(self, instructions_units):
        """
        instructions_units: list[list[str]]
            [['intersection', 'a red Coca-Cola sign', ...]]

        return: list[dict]
            [
                {
                    'input_ids': Tensor[num_units, L],
                    'attention_mask': Tensor[num_units, L]
                },
                ...
            ]
        """
        if instructions_units is None:
            return None

        tokenized_units = []
        for units in instructions_units:
            if units is None:
                tokenized_units.append(None)
                continue

            enc = self.tokenizer(
                units,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32,
                add_special_tokens=True,
            )

            tokenized_units.append({
                "input_ids": enc["input_ids"].to(self.model.device),
                "attention_mask": enc["attention_mask"].to(self.model.device),
            })

        return tokenized_units

    def prepare_inputs(self, episodes, target_positions, instructions_units: Optional[list[list[str]]] = None, assist_notices=None):
        """
        episodes: [[{}, ..., {}], [{}, ..., {}]] (batch_size = 2)
        episodes[0][0].keys() = dict_keys(['sensors', 'instruction', 'trajectory_dir', 'teacher_action', 'rgb', 'depth', 'rgb_record', 'depth_record'])
        self.data_args: DataArguments(data_path=None, lazy_preprocess=False, is_multimodal=False, image_grid_pinpoints=None, input_prompt=None, refine_prompt=True, mm_use_im_start_end=False)
        """
        inputs = []
        rot_to_targets = []
        # import pdb; pdb.set_trace()
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                instruction_units=instructions_units[i] if instructions_units is not None else None,
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)
        """
        batch.keys() = dict_keys(['input_ids', 'labels', 'attention_mask', 'images', 'prompts', 'historys', 'orientations', 'instructions_units'])
        """
        # import pdb; pdb.set_trace() # 检查 batch 内容，prompts 和 instructions_units 是不是文本形式
        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'instructions_units' not in k and 'images' not in k and 'historys' not in k}
        inputs_device['prompts'] = [item for item in batch['prompts']]
        inputs_device['instructions_units'] = self._tokenize_instruction_units(
            batch['instructions_units']
        )
        inputs_device['reset_memory'] = (len(episodes[0]) == 1)
        inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
        inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
        inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
        inputs_device['return_waypoints'] = True
        inputs_device['use_cache'] = False
        """
        inputs_device.keys() = dict_keys(['input_ids', 'labels', 'attention_mask', 'orientations', 'prompts', 'images', 'historys', 'return_waypoints', 'use_cache'])
        """
        return inputs_device, rot_to_targets

    def run_llm_model(self, inputs):
        waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs, None)
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints
    
    def eval(self):
        self.model.eval()
        self.traj_model.eval()
        
    def run(self, inputs, episodes, rot_to_targets):
        waypoints_llm_new = self.run_llm_model(inputs)
        refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
        return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        

    
import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized

def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    model_wrapper.eval()
    
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)
            # import pdb; pdb.set_trace() # 检查 batch_state 内容，尤其是 episodes 和 target_positions
            pbar.update(n=eval_env.batch_size)
            
            inputs, rot_to_targets = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions, batch_state.instructions_units)
            # import pdb; pdb.set_trace()
            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                # import pdb; pdb.set_trace() # 检查 inputs 是否包含所有 llava_llama_uav 所需的参数，数据类型是否匹配
                """
                inputs["instructions_units"][0] = 
                {'input_ids': tensor([[    1, 17686,     0,     0,     0,     0],
                        [    1,   263,  2654,  1559,     0,     0],
                        [    1, 17686,   411,   263,  2654,  1559],
                        [    1, 17686,     0,     0,     0,     0],
                        [    1, 14089,     0,     0,     0,     0],
                        [    1, 17686,     0,     0,     0,     0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0]], device='cuda:0')}
                """
                """
                inputs.keys() = dict_keys(['input_ids', 'labels', 'attention_mask', 'orientations', 'prompts', 'instructions_units', 'reset_memory', 'images', 'historys', 'return_waypoints', 'use_cache'])
                """
                refined_waypoints = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets) # key function
                eval_env.makeActions(refined_waypoints)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                
                batch_state.update_metric()
                
                assist_notices = batch_state.get_assist_notices()
                inputs, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions, batch_state.instructions_units, assist_notices)

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    
    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    model_args.batch_size = args.batchSize
    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    
    eval(model_wrapper=model_wrapper,
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)
    
    eval_env.delete_VectorEnvUtil()

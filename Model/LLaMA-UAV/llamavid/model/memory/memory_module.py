import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MemoryModule(nn.Module):
    def __init__(self, batch_size: int, memory_size = 1000, *args, **kwargs):
        super(MemoryModule, self).__init__(*args, **kwargs)
        self.memory_bank: Optional[torch.Tensor] = None # [B, num_frame, D]
        self.truncated_count: int = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.instructions_units: list[list[torch.Tensor]] = [] # [B][num_units] token
        self.retrieved_memories: list[list[int]] = [] # [B][retrieved_nums] int (frame_id)
        self.mem_progress: Optional[list[int]] = [] # [B] int (start from 0, means how many retrieved memories are fixed)
        self.retrieve_threshold = 0.7

        self.q_proj = nn.Linear(4096, 4096)
        self.k_proj = nn.Linear(4096, 4096)
        self.v_proj = nn.Linear(4096, 4096)
        self.output_proj = nn.Linear(4096, 4096)
        self.n_heads = 8
        self.d_head = 4096 // self.n_heads
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(4096)
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    
    def store_memory(self, features: torch.Tensor):
        """
        Store current memory into the memory bank.

        features: [B, S, D]
        """
        compressed_memory = features.mean(1, keepdim=True)
        compressed_memory = compressed_memory.detach()

        if self.memory_bank is None:
            self.memory_bank = compressed_memory
        else:
            self.memory_bank = torch.cat([self.memory_bank, compressed_memory], dim=1)
            if self.memory_bank.shape[1] > self.memory_size:
                self.truncated_count += 1
                print(f"Memory bank exceeded size {self.memory_size}, truncating {self.truncated_count} memories.")
                self.memory_bank = self.memory_bank[:, -self.memory_size:]

    def retrieve_memory(self, hidden_states: torch.Tensor):
        """
        retrieve and integrate memory for the current frame
        """
        if self.memory_bank is None:
            return hidden_states
        if self.instructions_units is None or len(self.instructions_units) == 0:
            return hidden_states
        
        def _retrieve():
            """
            retrieve corresponding memory for the latest 2 units
                unit_id 0       1       2       3       4
            mem_prog
                0       retr    retr
                1       fixed   retr    retr
                2       fixed   fixed   retr    retr
                3       fixed   fixed   fixed   retr    retr
                4       fixed   fixed   fixed   fixed   retr
            """
            for b in range(self.batch_size):
                if self.mem_progress[b] >= len(self.instructions_units[b]):
                    continue
                # retrieve memory for the current instruction unit
                start_frame = self.retrieved_memories[b][self.mem_progress[b] - 1] + 3 if self.mem_progress[b] > 0 else 0
                # retrieve the most similar frame for the current instruction unit
                num_candidate_frames = self.memory_bank.shape[1] - start_frame
                if num_candidate_frames <= 0:
                    continue
                unit = self.instructions_units[b][self.mem_progress[b] + 0].unsqueeze(0).expand(num_candidate_frames, -1) # [num_candidate_frames, D]
                cos = F.cosine_similarity(unit, self.memory_bank[b, start_frame:, :], dim=-1)  # [num_candidate_frames]
                max_score = ((cos.max().item() + 1) * 0.5)
                max_idx = cos.argmax().item()
                if max_score > self.retrieve_threshold:
                    if len(self.retrieved_memories[b]) == self.mem_progress[b]:
                        self.retrieved_memories[b].append(start_frame + max_idx)
                    else:
                        self.retrieved_memories[b][self.mem_progress[b]] = start_frame + max_idx

                # retrieve memory for the next instruction unit
                if len(self.retrieved_memories[b]) == self.mem_progress[b]: # if current instruction unit is not retrieved, then skip retrieving the next instruction unit
                    continue
                else: # current instruction unit is retrieved, then try to retrieve the next instruction unit
                    unit = self.instructions_units[b][self.mem_progress[b] + 1].unsqueeze(0).expand(num_candidate_frames, -1) if self.mem_progress[b] + 1 < len(self.instructions_units[b]) else None
                    if unit is not None:
                        cos = F.cosine_similarity(unit, self.memory_bank[b, start_frame:, :], dim=-1)  # [num_candidate_frames]
                        max_score = ((cos.max().item() + 1) * 0.5)
                        max_idx = cos.argmax().item()
                        if max_score > self.retrieve_threshold and self.mem_progress[b] < len(self.retrieved_memories[b]) and start_frame + max_idx > self.retrieved_memories[b][self.mem_progress[b]]:
                            self.mem_progress[b] += 1

        _retrieve() # retrieve corresponding memory for the latest 2 instruction units
        
        B, N_h, D = hidden_states.size()
        assert B == self.batch_size, "hidden_states.size(0) != memory_module.batch_size"
        retrieved_nums = [len(self.retrieved_memories[b]) for b in range(B)]

        Q = self.q_proj(hidden_states)  # [B, N_h, D]
        K, V = [], []
        for b in range(B):
            if retrieved_nums[b] != 0:
                assert self.memory_bank is not None, "Memory bank is None while retrieved_nums[b] != 0."
                K.append(self.k_proj(
                    self.memory_bank[b, self.retrieved_memories[b], :]
                )) # [retrieved_num_b, D]
                V.append(self.v_proj(
                    self.memory_bank[b, self.retrieved_memories[b], :]
                )) # [retrieved_num_b, D]
            else:
                K.append(torch.zeros(1, D, device=hidden_states.device, dtype=hidden_states.dtype)) # [1, D]
                V.append(torch.zeros(1, D, device=hidden_states.device, dtype=hidden_states.dtype)) # [1, D]
        
        # reshape for multi-head attention
        Q = Q.view(B, N_h, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N_h, Dh)
        for b in range(B):
            if retrieved_nums[b] != 0:
                K[b] = K[b].view(retrieved_nums[b], self.n_heads, self.d_head).transpose(0, 1)  # (H, N_r, Dh)
                V[b] = V[b].view(retrieved_nums[b], self.n_heads, self.d_head).transpose(0, 1)  # (H, N_r, Dh)
            else:
                K[b] = K[b].view(1, self.n_heads, self.d_head).transpose(0, 1)  # (H, 1, Dh)
                V[b] = V[b].view(1, self.n_heads, self.d_head).transpose(0, 1)  # (H, 1, Dh)

        attn_scores = [torch.matmul(Q[b], K[b].transpose(-2, -1)) / (self.d_head ** 0.5) for b in range(B)] # [B](H, N_h, N_r)
        attn_weights = []
        for scores, N_r in zip(attn_scores, retrieved_nums):
            if N_r == 0:
                attn_weights.append(torch.zeros_like(scores))
            else:
                attn_weights.append(F.softmax(scores, dim=-1)) # [B](H, N_h, N_r)
        attn_weights = [self.dropout(weights) for weights in attn_weights]

        retrieved_list = [torch.matmul(attn_weight, v) for attn_weight, v in zip(attn_weights, V)]  # [B](H, N_h, Dh)
        retrieved_list = [retrieved.transpose(0, 1).contiguous().view(N_h, D) for retrieved in retrieved_list]  # [B][N_h, D]
        retrieved_memories = torch.stack(retrieved_list, dim=0)  # [B, N_h, D]
        retrieved_memories = self.output_proj(retrieved_memories) * self.gate  # [B, N_h, D]
        integrated = self.norm(hidden_states + retrieved_memories)

        return integrated

    def reset_memory(self, instr: list[list[torch.Tensor]]):
        self.memory_bank = None
        self.truncated_count = 0
        self.instructions_units = instr
        self.retrieved_memories = []
        self.mem_progress = []

        for b in range(self.batch_size):
            self.retrieved_memories.append([]) # [B][]
            self.mem_progress.append(0) # [B]

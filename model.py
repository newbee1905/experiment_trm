import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class RMSNorm(nn.Module):
	"""
	Root Mean Square Layer Normalization Gemma zero centred variant.
	Reference: Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
	"""
	def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
		super().__init__()
		self.eps = eps
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float())
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()
		return output.type_as(x)

def get_rotary_sin_cos(dim, max_len, device):
	"""
	Generates RoPE embeddings.
	Reference: Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding.
	"""
	inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
	t = torch.arange(max_len, device=device).float()

	freqs = torch.einsum('i,j->ij', t, inv_freq)
	emb = torch.cat((freqs, freqs), dim=-1)

	return emb.cos(), emb.sin()

def apply_rotary_emb(x, cos, sin):
	assert x.ndim == 4 
	d = x.shape[3] // 2

	x1, x2 = x[..., :d], x[..., d:] 
	cos = cos[..., :d]
	sin = sin[..., :d]

	y1 = x1 * cos + x2 * sin 
	y2 = x1 * (-sin) + x2 * cos 

	out = torch.cat([y1, y2], 3)

	return out.to(x.dtype)

class MLP(nn.Module):
	def __init__(self, hidden_size, expansion_factor=4, dropout=0.0):
		super().__init__()
		self.c_fc = nn.Linear(hidden_size, hidden_size * expansion_factor, bias=False)
		self.c_gate = nn.Linear(hidden_size, hidden_size * expansion_factor, bias=False)
		self.c_proj = nn.Linear(hidden_size * expansion_factor, hidden_size, bias=False)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		h = self.c_fc(x)
		gate = self.c_gate(x)
		x = F.silu(h) * gate
		x = self.c_proj(self.dropout(x))

		return x

class MHA(nn.Module):
	def __init__(self, config, max_len=512):
		super().__init__()
		assert config.hidden_size % config.num_heads == 0

		self.d_model = config.hidden_size
		self.n_head = config.num_heads
		self.d_head = self.d_model // self.n_head
		self.use_rope = config.use_rope
		
		self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		
		# G1 Gate: Element-wise sigmoid gate applied to SDPA output
		# Reference: "Gated Attention for Large Language Models", arXiv:2505.06708
		self.g1_gate = nn.Linear(self.d_model, self.d_model, bias=False)
		
		self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
		
		if self.use_rope:
			self.max_len = max_len
			self.register_buffer('cos_cached', None, persistent=False)
			self.register_buffer('sin_cached', None, persistent=False)
			
		self.dropout = nn.Dropout(config.dropout)

	def _update_rope_cache(self, seq_len, device):
		if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:
			alloc_len = max(self.max_len, seq_len)
			cos, sin = get_rotary_sin_cos(self.d_head, alloc_len, device)
			self.cos_cached = cos
			self.sin_cached = sin

	def forward(self, x, mask=None):
		bsz, seq_len, _ = x.shape
		
		q = self.q_proj(x).view(bsz, seq_len, self.n_head, self.d_head)
		k = self.k_proj(x).view(bsz, seq_len, self.n_head, self.d_head)
		v = self.v_proj(x).view(bsz, seq_len, self.n_head, self.d_head)
		
		if self.use_rope:
			self._update_rope_cache(seq_len, x.device)
			cos = self.cos_cached[:seq_len].view(1, seq_len, 1, self.d_head)
			sin = self.sin_cached[:seq_len].view(1, seq_len, 1, self.d_head)
			q = apply_rotary_emb(q, cos, sin)
			k = apply_rotary_emb(k, cos, sin)

		# Transpose for SDPA: [bsz, n_head, seq_len, d_head]
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		
		# SDPA output
		output = F.scaled_dot_product_attention(
			q, k, v, 
			attn_mask=mask,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False 
		)
		
		# Transpose back: [bsz, seq_len, n_head, d_head]
		output = output.transpose(1, 2)
		
		# G1 Gating Mechanism
		# The paper uses the pre-norm input X to compute the gate
		# Gate = Sigmoid(X * W_theta)
		gate_score = self.g1_gate(x) # [bsz, seq_len, d_model]
		gate = torch.sigmoid(gate_score)
		
		# Reshape for element-wise multiplication with heads
		# [bsz, seq_len, d_model] -> [bsz, seq_len, n_head, d_head]
		gate = gate.view(bsz, seq_len, self.n_head, self.d_head)
		
		output = output * gate
		
		output = output.contiguous().view(bsz, seq_len, -1)
		return self.out_proj(output)

@dataclass
class ACTConfig:
	vocab_size: int = 11 # 0-9 for Sudoku + padding if needed
	hidden_size: int = 384
	num_heads: int = 6
	expansion: int = 4
	dropout: float = 0.1
	max_steps: int = 16 # ACT recursion limit
	use_rope: bool = True
	act_threshold: float = 0.99 
	act_ponder_penalty: float = 0.01

class ReasoningBlock(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.norm1 = RMSNorm(config.hidden_size)
		self.attn = MHA(config, max_len=81) # Sudoku is fixed 81
		self.norm2 = RMSNorm(config.hidden_size)
		self.mlp = MLP(config.hidden_size, config.expansion, config.dropout)

	def forward(self, x):
		h = self.norm1(x)
		x = x + self.attn(h)

		h = self.norm2(x)
		x = x + self.mlp(h)

		return x

class SudokuACT(nn.Module):
	"""
	Adaptive Computation Time model for Sudoku.
	Reference: Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.
	"""
	def __init__(self, config: ACTConfig):
		super().__init__()
		self.config = config
		
		self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
		self.block = ReasoningBlock(config)
		
		self.halting_linear = nn.Linear(config.hidden_size, 1)
		
		self.head = nn.Linear(config.hidden_size, config.vocab_size)
		
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, x, targets=None):
		bsz, seq_len = x.shape
		
		state = self.embedding(x) # [bsz, seq_len, d_model]
		
		# ACT Variables
		# Accumulated output
		out_accum = torch.zeros_like(state)

		# Accumulated halting probability
		halt_accum = torch.zeros(bsz, seq_len, 1, device=x.device) 
		# Ponder cost accumulator (steps taken)
		ponder_cost = torch.zeros(bsz, seq_len, 1, device=x.device)
		# Tracking who has halted
		still_running = torch.ones(bsz, seq_len, 1, device=x.device, dtype=torch.bool)
		
		steps_taken = 0
		
		# Recursive Reason Loop
		for n in range(self.config.max_steps):
			if not still_running.any():
				break
			
			steps_taken += 1
			
			state = self.block(state)
			p_halt = torch.sigmoid(self.halting_linear(state))
			
			# Mask logic for ACT
			# If we already halted, probability of halting now is 0 effectively (for accumulation)
			# but we use the remainders mechanism from the paper.
			# R(t) = 1 - sum(h_k) for k < t
			remainder = 1.0 - halt_accum
			
			# If (accum + p) > 1, we must stop and take the remainder
			# halt_mask checks if we overshoot threshold 1.0 (epsilon used for safety)
			is_final_step = (halt_accum + p_halt) >= self.config.act_threshold
			
			# If it's the max step, force halt
			if n == self.config.max_steps - 1:
				is_final_step = torch.ones_like(is_final_step, dtype=torch.bool)
			
			# effective p_halt for this step
			p_use = torch.where(is_final_step, remainder, p_halt)
			
			# Apply only to those still running
			p_use = torch.where(still_running, p_use, torch.zeros_like(p_use))
			
			# Accumulate Output: out = sum(p_t * state_t)
			out_accum = out_accum + (p_use * state)
			
			# Update accumulators
			halt_accum = halt_accum + p_use
			ponder_cost = ponder_cost + still_running.float() # Add 1 for every step running
			
			# Update running mask
			# If we hit final step, we are no longer running
			still_running = still_running & (~is_final_step)
			
		# Final Projection
		logits = self.head(out_accum) # [B, L, Vocab]
		
		if targets is not None:
			# Classification Loss
			loss_ce = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-100)
			
			# Ponder Loss (encourage fewer steps)
			# Standard ACT loss: Mean(ponder_cost + remainder)
			# ponder_cost tracked whole steps.
			loss_ponder = self.config.act_ponder_penalty * ponder_cost.mean()
			
			total_loss = loss_ce + loss_ponder
			return logits, total_loss, steps_taken
			
		return logits, None, steps_taken

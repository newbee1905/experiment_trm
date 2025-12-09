import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import os
import math
import random
import copy
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import all_reduce, ReduceOp
import numpy as np
from dotenv import load_dotenv
import time

from datasets import load_dataset
from muon import MuonWithAuxAdam
import wandb

from model import SudokuACT, ACTConfig

load_dotenv()

class SudokuDataset(Dataset):
	def __init__(self, split="train", num_val_samples=10000, max_train_samples=None):
		print(f"Loading Sudoku dataset (split={split})...")
		full_ds = load_dataset("sapientinc/sudoku-extreme", split="train")

		total_size = len(full_ds)
		val_start_idx = total_size - num_val_samples

		if split == "train":
			self.augment = True
			if max_train_samples:
				stop_idx = min(max_train_samples, val_start_idx)
				self.dataset = full_ds.select(range(stop_idx))
			else:
				self.dataset = full_ds.select(range(val_start_idx))
		elif split == "val":
			self.augment = False
			self.dataset = full_ds.select(range(val_start_idx, total_size))
		
		self.split = split
		print(f"Loaded {len(self.dataset)} samples for {split}.")

	def __len__(self):
		return len(self.dataset)

	def augment_sudoku(self, puzzle, solution):
		"""
		Applies Sudoku-preserving symmetries:
		1. Permute digits (1-9).
		2. Permute row bands (0-2, 3-5, 6-8).
		3. Permute rows within bands.
		4. Permute col bands.
		5. Permute cols within bands.
		6. Transpose.
		"""
		p = puzzle.numpy().reshape(9, 9)
		s = solution.numpy().reshape(9, 9)
		
		# Digit Permutation (1-9 mapping)
		perm = np.random.permutation(9) + 1
		mapping = {i+1: perm[i] for i in range(9)}
		mapping[0] = 0 # Empty cell stays empty
		
		# Create a lookup table: index 0 -> 0, index i -> mapping[i]
		lookup = np.zeros(10, dtype=int)
		for k, v in mapping.items():
			lookup[k] = v
		
		p = lookup[p]
		s = lookup[s]
		
		if random.random() < 0.5:
			p = p.T
			s = s.T
			
		# Permute Row Bands (chunks of 3 rows)
		if random.random() < 0.5:
			band_perm = np.random.permutation(3)

			new_rows = []
			for b in band_perm:
				new_rows.extend(range(b*3, (b+1)*3))

			p = p[new_rows, :]
			s = s[new_rows, :]
			
		# Permute Rows within Bands
		for b in range(3):
			if random.random() < 0.5:
				row_perm = np.random.permutation(3) + b*3
				p[b*3:(b+1)*3, :] = p[row_perm, :]
				s[b*3:(b+1)*3, :] = s[row_perm, :]
		
		# Permute Col Bands
		if random.random() < 0.5:
			band_perm = np.random.permutation(3)
			new_cols = []

			for b in band_perm:
				new_cols.extend(range(b*3, (b+1)*3))

			p = p[:, new_cols]
			s = s[:, new_cols]
			
		# Permute Cols within Bands
		for b in range(3):
			if random.random() < 0.5:
				col_perm = np.random.permutation(3) + b*3
				p[:, b*3:(b+1)*3] = p[:, col_perm]
				s[:, b*3:(b+1)*3] = s[:, col_perm]
				
		return torch.from_numpy(p.flatten()), torch.from_numpy(s.flatten())
	
	def __getitem__(self, idx):
		item = self.dataset[idx]
		
		quiz_str = item['question']
		sol_str = item['answer']
		
		# '.' becomes 0, digits become ints
		puzzle_lst = []
		for c in quiz_str:
			if c == '.':
				puzzle_lst.append(0)
			else:
				puzzle_lst.append(int(c))
				
		solved_lst = [int(c) for c in sol_str]
		
		puzzle = torch.tensor(puzzle_lst, dtype=torch.long)
		solved = torch.tensor(solved_lst, dtype=torch.long)
		
		if self.augment:
			puzzle, solved = self.augment_sudoku(puzzle, solved)
		
		return puzzle, solved

class TrainingConfig:
	def __init__(self):
		self.lr = 5e-4 
		self.weight_decay = 0.01 
		
		self.batch_size = 192 
		self.gradient_accumulation_steps = 1
		
		self.max_steps = 80000 
		
		self.save_every = 2000
		self.grad_clip = 1.0 

class GlobalConfig:
	def __init__(self):
		self.training = TrainingConfig()
		self.experiment_name = "sudoku_act_real_data"
		self.wandb_project = "sudoku_act"

class Trainer:
	def __init__(self, model, config, train_loader, val_loader=None):
		self.config = config

		if not dist.is_initialized():
			dist.init_process_group(backend="nccl")

		self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
		self.global_rank = int(os.environ.get("RANK", 0))
		self.world_size = int(os.environ.get("WORLD_SIZE", 1))
		
		self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
		if torch.cuda.is_available():
			torch.cuda.set_device(self.device)
		
		self.model = model.to(self.device)
		
		if self.world_size > 1:
			self.model = DDP(self.model, device_ids=[self.local_rank])

		# Configure optimizers
		self.optimizer = self.configure_optimizers()
		
		self.train_loader = train_loader
		self.val_loader = val_loader
		
		# Initialize wandb (only on rank 0)
		if self.global_rank == 0:
			wandb_key = os.getenv("WANDB_API_KEY")
			if wandb_key:
				wandb.login(key=wandb_key)

			wandb.init(
				project=config.wandb_project,
				name=config.experiment_name,
				config=config.training.__dict__
			)
		
	def configure_optimizers(self):
		"""
		Configures the optimizer(s).
		- Separates parameters into Decay (weights) and No-Decay (bias/norm) groups.
		- If World Size > 1: Uses Muon for internal 2D weights and AdamW for others.
		- If World Size == 1: Uses AdamW for everything.
		"""

		body_params = list(raw_model.block.parameters())
		body_param_ids = set(id(p) for p in body_params)
		nonbody_params = [p for p in raw_model.parameters() if id(p) not in body_param_ids]
		
		# Categorize based on dimensions for Muon compatibility and Weight Decay logic
		# Muon targets >= 2D params inside the body
		muon_candidates = [p for p in body_params if p.ndim >= 2]
		
		# Remaining body params (1D: biases, norms) -> No Decay
		body_nodecay = [p for p in body_params if p.ndim < 2]
		
		# Non-body params split by dimension
		nonbody_decay = [p for p in nonbody_params if p.ndim >= 2] # Embeddings, Heads
		nonbody_nodecay = [p for p in nonbody_params if p.ndim < 2] # Biases
		
		# Group 1: No Decay (Biases, Norms from everywhere)
		# Weight decay strictly 0.0
		no_decay_group = {
			'params': body_nodecay + nonbody_nodecay,
			'weight_decay': 0.0,
			'use_muon': False,
			'lr': self.config.training.lr
		}
		
		# Group 2: Decay (Standard AdamW weights)
		# Weights outside the body (or all weights if not using Muon)
		adam_decay_group = {
			'params': nonbody_decay,
			'weight_decay': self.config.training.weight_decay,
			'use_muon': False,
			'lr': self.config.training.lr
		}
		
		if self.world_size > 1:
			if self.global_rank == 0:
				print("Distributed training detected: Using Muon + Aux AdamW")
				
			# Group 3: Muon (Body Matrix Weights)
			muon_group = {
				'params': muon_candidates,
				'weight_decay': self.config.training.weight_decay,
				'use_muon': True,
				'lr': 0.02 # Muon default LR
			}
			
			param_groups = [muon_group, adam_decay_group, no_decay_group]
			return MuonWithAuxAdam(param_groups)
			
		else:
			if self.global_rank == 0:
				print("Single GPU detected: Using AdamW")
				
			adam_decay_group['params'].extend(muon_candidates)
			
			return optim.AdamW([adam_decay_group, no_decay_group],  betas=(0.9, 0.95))

	def _dist_mean(self, tensor):
		if self.world_size > 1:
			all_reduce(tensor, op=ReduceOp.SUM)
			tensor /= self.world_size
		return tensor

	def save_checkpoint(self, step, loss):
		if self.global_rank != 0: return
		
		raw_model = self.model.module if isinstance(self.model, DDP) else self.model
		
		save_path = f"checkpoints/{self.config.experiment_name}"
		os.makedirs(save_path, exist_ok=True)
		
		torch.save({
			'step': step,
			'model_state_dict': raw_model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss': loss,
		}, f"{save_path}/checkpoint_{step}.pt")
		tqdm.write(f"Saved checkpoint to {save_path}/checkpoint_{step}.pt")

	@torch.no_grad()
	def evaluate(self, loader):
		if loader is None: return {}
		self.model.eval()
		total_loss = 0.0
		steps = 0
		total_correct = 0
		total_cells = 0
		
		max_val_batches = 100
		
		pbar = tqdm(loader, desc="Validating", disable=self.global_rank != 0, leave=False)
		
		for i, batch in enumerate(pbar):
			if i >= max_val_batches:
				break
			
			inputs, targets = [b.to(self.device) for b in batch]
			
			logits, loss, _ = self.model(inputs, targets=targets)
			
			total_loss += loss.item()
			
			preds = torch.argmax(logits, dim=-1)
			
			mask = targets != -100
			correct = (preds == targets) & mask
			total_correct += correct.sum().item()
			total_cells += mask.sum().item()
			steps += 1
			
		avg_loss = torch.tensor(total_loss / steps, device=self.device)
		avg_loss = self._dist_mean(avg_loss)
		
		accuracy = total_correct / total_cells if total_cells > 0 else 0
		
		self.model.train()
		return {"loss": avg_loss.item(), "accuracy": accuracy}

	def fit(self):
		self.model.train()
		step = 0
		max_steps = self.config.training.max_steps
		accum_steps = self.config.training.gradient_accumulation_steps
		grad_clip = self.config.training.grad_clip
		
		def cycle(loader):
			while True:
				for x in loader:
					yield x
		
		data_iter = cycle(self.train_loader)
		pbar = tqdm(range(max_steps), desc="Training", disable=self.global_rank != 0)
		
		current_loss = 0.0
		avg_act_steps = 0.0
		
		while step < max_steps:
			step_loss = 0.0

			for _ in range(accum_steps):
				inputs, targets = [b.to(self.device) for b in next(data_iter)]
				
				logits, loss, act_steps = self.model(inputs, targets=targets)
				
				loss = loss / accum_steps
				loss.backward()
				step_loss += loss.item()
				if isinstance(act_steps, torch.Tensor):
					avg_act_steps = act_steps.item()
				else:
					avg_act_steps = act_steps

			# Gradient Clipping
			if grad_clip > 0.0:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
				
			self.optimizer.step()
			self.optimizer.zero_grad()
			step += 1
			current_loss = step_loss * accum_steps
			
			if step % 10 == 0:
				pbar.set_postfix({
					'loss': f"{current_loss:.4f}",
					'steps': f"{avg_act_steps:.2f}"
				})

				# Log to wandb
				if self.global_rank == 0:
					wandb.log({
						"train/loss": current_loss,
						"train/act_steps": avg_act_steps,
						"train/step": step,
						"train/lr": self.optimizer.param_groups[0]['lr']
					})
			
			pbar.update(1)

			if step % self.config.training.save_every == 0:
				val_metrics = self.evaluate(self.val_loader)
				if self.global_rank == 0:
					tqdm.write(f"\nStep {step} | Val Loss: {val_metrics.get('loss', 0):.4f} | Acc: {val_metrics.get('accuracy', 0):.4f}")

					# Log validation metrics
					wandb.log({
						"val/loss": val_metrics.get('loss', 0),
						"val/accuracy": val_metrics.get('accuracy', 0),
						"val/step": step
					})
				self.save_checkpoint(step, current_loss)
		
		pbar.close()
		if self.global_rank == 0:
			wandb.finish()

if __name__ == "__main__":
	# Config matching TRM Paper
	act_config = ACTConfig(
		hidden_size=512, 
		num_heads=8,
		max_steps=16, 
		act_ponder_penalty=0.01 
	)
	global_config = GlobalConfig()
	
	# Data 
	train_dataset = SudokuDataset(split="train", max_train_samples=100000) 
	val_dataset = SudokuDataset(split="val", num_val_samples=1000)
	
	train_loader = DataLoader(
		train_dataset, 
		batch_size=global_config.training.batch_size, 
		shuffle=True, 
		num_workers=16, 
		pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset, 
		batch_size=global_config.training.batch_size, 
		shuffle=False, 
		num_workers=16
	)
	
	# Model
	model = SudokuACT(act_config)
	
	if int(os.environ.get("RANK", 0)) == 0:
		print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
	
	# Trainer
	trainer = Trainer(model, global_config, train_loader, val_loader)
	try:
		trainer.fit()
	finally:
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()

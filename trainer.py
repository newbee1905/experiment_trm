import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
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
	def __init__(self, split="train", num_val_samples=10_000, max_train_samples=100_000):
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
		p = puzzle.view(9, 9)
		s = solution.view(9, 9)

		# Random mapping of digits 1-9
		if random.random() < 0.5:
			perm = torch.randperm(9) + 1
			mapping = torch.zeros(10, dtype=torch.long)
			mapping[1:] = perm
			p = mapping[p]
			s = mapping[s]

		if random.random() < 0.5:
			p = p.t()
			s = s.t()

		# Permute Bands (Rows 0-2 vs 3-5 vs 6-8)
		if random.random() < 0.5:
			band_perm = torch.randperm(3)
			idx = torch.cat([torch.arange(b*3, (b+1)*3) for b in band_perm])
			p = p[idx]
			s = s[idx]

		# Permute Rows WITHIN Bands 
		if random.random() < 0.5:
			p = p.view(3, 3, 9)
			s = s.view(3, 3, 9)

			# Generate 3 independent permutations for the rows
			for b in range(3):
				row_perm = torch.randperm(3)
				p[b] = p[b][row_perm]
				s[b] = s[b][row_perm]

			p = p.view(9, 9)
			s = s.view(9, 9)

		# Permute Cols (similar to bands)
		if random.random() < 0.5:
			band_perm = torch.randperm(3)
			idx = torch.cat([torch.arange(b*3, (b+1)*3) for b in band_perm])
			p = p[:, idx]
			s = s[:, idx]

		# Permute Cols WITHIN Bands
		if random.random() < 0.5:
			p = p.view(9, 3, 3)
			s = s.view(9, 3, 3)

			for b in range(3):
				col_perm = torch.randperm(3)
				p[:, b] = p[:, b][:, col_perm]
				s[:, b] = s[:, b][:, col_perm]

			p = p.view(9, 9)
			s = s.view(9, 9)
			
		return p.flatten(), s.flatten()
	
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
		
		# self.batch_size = 192 
		# self.gradient_accumulation_steps = 1
		self.batch_size = 16
		self.gradient_accumulation_steps = 1
		
		self.num_epochs = 60000
		
		self.save_every = 5000
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
		self.scaler = torch.cuda.amp.GradScaler(enabled=True)
		
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

		body_params = list(self.model.parameters())
		body_param_ids = set(id(p) for p in body_params)
		nonbody_params = [p for p in self.model.parameters() if id(p) not in body_param_ids]
		
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
		
		save_path = f"checkpoints/{self.config.experiment_name}"
		os.makedirs(save_path, exist_ok=True)
		
		torch.save({
			'step': step,
			'model_state_dict': self.model.state_dict(),
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
		
		total_cells = 0
		correct_cells = 0
		
		total_puzzles = 0
		correct_puzzles = 0
		
		max_val_batches = 100
		
		pbar = tqdm(loader, desc="Validating", disable=self.global_rank != 0, leave=False)
		
		for i, batch in enumerate(pbar):
			if i >= max_val_batches:
				break
			
			inputs, targets = [b.to(self.device) for b in batch]
			
			logits, loss, _ = self.model(inputs, targets=targets)
			total_loss += loss.item()
			
			# shape: [batch_size, 81, vocab_size] -> [batch_size, 81]
			preds = torch.argmax(logits, dim=-1)
			
			# Cell-wise Accuracy
			mask = targets != -100
			
			# Compare only valid cells
			cell_hits = (preds == targets) & mask
			correct_cells += cell_hits.sum().item()
			total_cells += mask.sum().item()
			
			# Exact Match Accuracy (Whole Puzzle)
			row_hits = (preds == targets).all(dim=1)
			correct_puzzles += row_hits.sum().item()
			total_puzzles += inputs.size(0)
			
			steps += 1
			
		# Aggregate across GPUs
		avg_loss = torch.tensor(total_loss / steps, device=self.device)
		total_cells_t = torch.tensor(total_cells, device=self.device)
		correct_cells_t = torch.tensor(correct_cells, device=self.device)
		total_puzzles_t = torch.tensor(total_puzzles, device=self.device)
		correct_puzzles_t = torch.tensor(correct_puzzles, device=self.device)

		if self.world_size > 1:
			all_reduce(avg_loss, op=ReduceOp.SUM)
			avg_loss /= self.world_size
			
			all_reduce(total_cells_t, op=ReduceOp.SUM)
			all_reduce(correct_cells_t, op=ReduceOp.SUM)
			all_reduce(total_puzzles_t, op=ReduceOp.SUM)
			all_reduce(correct_puzzles_t, op=ReduceOp.SUM)

		cell_acc = correct_cells_t.item() / total_cells_t.item() if total_cells_t.item() > 0 else 0
		exact_acc = correct_puzzles_t.item() / total_puzzles_t.item() if total_puzzles_t.item() > 0 else 0
		
		self.model.train()
		return {
			"loss": avg_loss.item(), 
			"accuracy": cell_acc,	
			"exact_accuracy": exact_acc,
		}

	def fit(self):
		steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
		total_steps = self.config.training.num_epochs * steps_per_epoch
		
		if self.global_rank == 0:
			print(f"Training for {self.config.training.num_epochs} epochs.")
			print(f"Total optimization steps: {total_steps}")
			print(f"Steps per epoch: {steps_per_epoch}")

		self.model.train()
		global_step = 0
		
		epoch_pbar = tqdm(range(self.config.training.num_epochs), desc="Training", disable=self.global_rank != 0)
		
		for epoch in epoch_pbar:
			if isinstance(self.train_loader.sampler, torch.utils.data.DistributedSampler):
				self.train_loader.sampler.set_epoch(epoch)
			
			for batch_idx, batch in enumerate(self.train_loader):
				inputs, targets = [b.to(self.device) for b in batch]

				with torch.autocast(device_type="cuda", dtype=torch.float16):
					logits, loss, act_steps = self.model(inputs, targets=targets)
					loss = loss / self.config.training.gradient_accumulation_steps

				self.scaler.scale(loss).backward()
				
				current_act_steps = act_steps.item() if isinstance(act_steps, torch.Tensor) else act_steps

				if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
					if self.config.training.grad_clip > 0.0:
						torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
					
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.optimizer.zero_grad()
					global_step += 1
					
					current_loss = loss.item() * self.config.training.gradient_accumulation_steps
					
					if self.global_rank == 0:
						epoch_pbar.set_postfix({
							'step': global_step,
							'loss': f"{current_loss:.4f}",
							'act': f"{current_act_steps:.1f}"
						})

						# Log to WandB every step (or every N steps if preferred)
						wandb.log({
							"train/loss": current_loss,
							"train/act_steps": current_act_steps,
							"train/global_step": global_step,
							"train/epoch": epoch + (batch_idx / steps_per_epoch), # fractional epoch
							"train/lr": self.optimizer.param_groups[0]['lr']
						})

			if (epoch + 1) % self.config.training.save_every == 0:
				val_metrics = self.evaluate(self.val_loader)
				
				if self.global_rank == 0:
					tqdm.write(
						f"Epoch {epoch+1} | "
						f"Loss: {val_metrics.get('loss', 0):.4f} | "
						f"Cell Acc: {val_metrics.get('accuracy', 0):.4f} | "
						f"Exact Match: {val_metrics.get('exact_accuracy', 0):.4f}"
					)
					
					wandb.log({
						"val/loss": val_metrics.get('loss', 0),
						"val/accuracy": val_metrics.get('accuracy', 0),
						"val/exact_accuracy": val_metrics.get('exact_accuracy', 0),
						"val/epoch": epoch + 1
					})
					
					self.save_checkpoint(epoch, val_metrics.get('loss', 0))

		epoch_pbar.close()
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

	is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
	
	# Data 
	train_dataset = SudokuDataset(split="train", max_train_samples=100_000) 
	val_dataset = SudokuDataset(split="val", num_val_samples=1000)

	if is_ddp:
		sampler = DistributedSampler(train_dataset)
		shuffle = False 
	else:
		sampler = None
		shuffle = True
	
	train_loader = DataLoader(
		train_dataset, 
		batch_size=global_config.training.batch_size, 
		shuffle=True, 
		num_workers=10, 
		sampler=sampler,
		persistent_workers=True,
		pin_memory=True,
	)

	val_loader = DataLoader(
		val_dataset, 
		batch_size=global_config.training.batch_size, 
		shuffle=False, 
		num_workers=10,
		pin_memory=True,
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

# block_bae_pl/datamodule.py (final complete version)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
import json
from pathlib import Path
import pytorch_lightning as pl
import os
import h5py
import numpy as np
import logging

log = logging.getLogger(__name__)

class BlockCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
        padded_batch = pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
        return {'src': padded_batch, 'len': lengths}

class HDF5IterableDataset(IterableDataset):
    def __init__(self, h5_path: Path, vocab_path: Path, indices: list, max_len: int, buffer_size: int = 1_000_000, seed: int = 42):
        super().__init__()
        self.h5_path = h5_path
        self.vocab_path = vocab_path
        self.max_len = max_len
        self.buffer_size = buffer_size
        self.seed = seed
        self.epoch = 0
        self.all_indices = np.array(indices, dtype=np.int64)

        with open(self.vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab_data['itos_lst'])}
        self.sos_id = self.token_to_id['<sos>']
        self.eos_id = self.token_to_id['<eos>']
        self.pad_id = self.token_to_id['<pad>']
        self.unk_id = self.token_to_id['<unk>']
        
        np.random.default_rng(self.seed).shuffle(self.all_indices)
        log.info(
            "HDF5IterableDataset initialised with %d indices (buffer=%d).",
            len(self.all_indices),
            self.buffer_size,
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        log.info(f"HDF5IterableDataset epoch set to {self.epoch}")

    def __iter__(self):
        seed = self.seed + self.epoch
        rng = np.random.default_rng(seed)
        shuffled_indices = self.all_indices.copy()
        rng.shuffle(shuffled_indices)

        with h5py.File(self.h5_path, "r", libver="latest", swmr=True) as h5_file:
            tokens_dset = h5_file["tokens"]
            indices_dset = h5_file["indices"]
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                indices_to_process = shuffled_indices
            else:
                per_worker = int(np.ceil(len(shuffled_indices) / worker_info.num_workers))
                start = worker_info.id * per_worker
                end = min(start + per_worker, len(shuffled_indices))
                indices_to_process = shuffled_indices[start:end]

            for offset in range(0, len(indices_to_process), self.buffer_size):
                chunk = indices_to_process[offset : offset + self.buffer_size].copy()
                rng.shuffle(chunk)
                for global_idx in chunk:
                    try:
                        start_pos, length = indices_dset[global_idx]
                        tokens_np = tokens_dset[start_pos : start_pos + length]
                        if len(tokens_np) > self.max_len:
                            tokens_np = np.concatenate([tokens_np[: self.max_len - 1], [self.eos_id]])
                        yield torch.from_numpy(tokens_np.astype(np.int64))
                    except Exception as exc:
                        log.error("failed to load sample %d: %s", global_idx, exc)
                        continue
    def __len__(self):
        return int(len(self.all_indices))

class HDF5BlockDataset(Dataset):
    def __init__(self, h5_path: Path, vocab_path: Path, indices: list, max_len: int):
        super().__init__()
        self.h5_path = h5_path
        self.vocab_path = vocab_path
        self.indices = indices
        self.max_len = max_len

        with open(self.vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab_data['itos_lst'])}
        self.eos_id = self.token_to_id['<eos>']
        
        self.h5_file = None
        self.tokens_dset = None
        self.indices_dset = None
    
    def _init_worker(self):
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.tokens_dset = self.h5_file['tokens']
        self.indices_dset = self.h5_file['indices']
        
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.h5_file is None: self._init_worker()
        global_index = self.indices[index]
        start_pos, length = self.indices_dset[global_index]
        token_ids_np = self.tokens_dset[start_pos : start_pos + length]
        if len(token_ids_np) > self.max_len:
            eos_token = np.array([self.eos_id])
            token_ids_np = np.concatenate([token_ids_np[:self.max_len - 1], eos_token])
        return torch.tensor(token_ids_np.astype(np.int64))

class BlockDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4,
                 val_split: float = 0.01, seed: int = 42, max_len: int = 128,
                 iterable_buffer_size: int = 200_000):
        super().__init__()
        self.save_hyperparameters()
        self.train_indices = None

    def prepare_data(self):
        frag_vocab_path = Path(self.hparams.data_dir) / "vocabulary.json"
        h5_path = Path(self.hparams.data_dir) / "sequences.h5"
        if not frag_vocab_path.exists() or not h5_path.exists():
            raise FileNotFoundError(f"Data files not found in {self.hparams.data_dir}.")

    def setup(self, stage: str = None):
        frag_vocab_path = Path(self.hparams.data_dir) / "vocabulary.json"
        h5_path = Path(self.hparams.data_dir) / "sequences.h5"
        
        with open(frag_vocab_path, 'r') as f:
            vocab_data = json.load(f)
        with h5py.File(h5_path, 'r') as hf:
            dataset_size = len(hf['indices'])

        token_to_id = {token: i for i, token in enumerate(vocab_data['itos_lst'])}
        self.pad_id = token_to_id['<pad>']
        self.vocab_size = len(token_to_id)

        all_indices = np.arange(dataset_size)
        rng = np.random.default_rng(self.hparams.seed)
        rng.shuffle(all_indices)
        
        val_size = int(self.hparams.val_split * dataset_size)
        train_size = dataset_size - val_size
        self.train_indices = all_indices[:train_size].tolist()
        val_indices = all_indices[train_size:].tolist()
        
        self.train_dataset = HDF5IterableDataset(h5_path, frag_vocab_path, self.train_indices, max_len=self.hparams.max_len, buffer_size=self.hparams.iterable_buffer_size)
        self.val_dataset = HDF5BlockDataset(h5_path, frag_vocab_path, val_indices, max_len=self.hparams.max_len)
        self.collator = BlockCollator(self.pad_id)
        
        log.info(f"DataModule setup complete. Vocab size: {self.vocab_size}, Pad ID: {self.pad_id}.")
        log.info(f"Dataset split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, collate_fn=self.collator, 
                          pin_memory=True, persistent_workers=self.hparams.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, collate_fn=self.collator, 
                          pin_memory=True, persistent_workers=self.hparams.num_workers > 0)
    
    def train_eval_dataloader(self):
        if self.train_indices is None:
            log.error("train_indices not set in setup(). Cannot create train_eval_dataloader.")
            return DataLoader([])
        
        h5_path = Path(self.hparams.data_dir) / "sequences.h5"
        frag_vocab_path = Path(self.hparams.data_dir) / "vocabulary.json"

        train_eval_dataset = HDF5BlockDataset(
            h5_path=h5_path,
            vocab_path=frag_vocab_path,
            indices=self.train_indices,
            max_len=self.hparams.max_len
        )

        return DataLoader(
            train_eval_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator
        )
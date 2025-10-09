#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import torch
import symusic
import pickle

class MIREXCustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        quantized_dir: Path, 
        tokenizer, 
        max_pitch_offset: int = 0,
        max_seq_len: int = 1024
        ):

        super().__init__()
        self._pkl_files = list(quantized_dir.glob("**/*.pkl"))
        self._tokenizer = tokenizer
        self._num_prompt_measures = 4
        self._num_completion_measures = 12
        self._max_pitch_offset = max_pitch_offset
        self._max_seq_len = max_seq_len
        
        if not self._pkl_files:
            raise ValueError(f"No .pkl files found in {quantized_dir}")
        
        print(f"Loaded {len(self._pkl_files)} quantized track files")
    
    def __len__(self):
        return len(self._pkl_files)
    
    def __getitem__(self, idx: int):
        pkl_path = self._pkl_files[idx]
        
        with open(pkl_path, "rb") as f:
            track = pickle.load(f)
        
        score = symusic.Score(ttype=symusic.TimeUnit.tick)
        score.tracks.append(track)
        
        encoding = self._tokenizer.encode(score)[0]
        token_ids = np.array(encoding.ids, dtype=np.int32)
        
        # find all bar boundaries
        bar_starts = np.where(token_ids == self._tokenizer.vocab["Bar_None"])[0]
        
        # need at least 17 bars to select a 16-bar window
        if len(bar_starts) < 17:
            # if not enough bars, just use whatever we have
            sample = token_ids
        else:
            # try up to 10 times to find a 16-bar window with > 100 tokens
            for _ in range(10):
                selected_bar_start = np.random.randint(0, len(bar_starts) - 16)
                sample_start = bar_starts[selected_bar_start]
                sample_end = bar_starts[selected_bar_start + 16]
                sample = token_ids[sample_start:sample_end]
                
                if len(sample) > 100:
                    break
        
        input_ids = torch.from_numpy(sample.copy()).long()
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        labels = input_ids.clone()[1:]
        input_ids = input_ids[:-1]
        
        tgt_len = min(len(labels), len(input_ids), self._max_seq_len)
        labels = labels[:tgt_len]
        input_ids = input_ids[:tgt_len]
        attention_mask = attention_mask[:tgt_len]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    import miditok
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        max_len = max(item['input_ids'].shape[0] for item in batch)
    
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
    
        for item in batch:
            seq_len = item['input_ids'].shape[0]
            pad_len = max_len - seq_len
        
            input_ids = torch.nn.functional.pad(
                item['input_ids'], 
                (0, pad_len), 
                value=0
                )
        
            attention_mask = torch.nn.functional.pad(
                item['attention_mask'], 
                (0, pad_len), 
                value=0
                )
        
            labels = torch.nn.functional.pad(
                item['labels'], 
                (0, pad_len), 
                value=-100
                )   
        
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
    
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            }

    # create tokenizer with simplified REMI config
    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
        special_tokens=["PAD", "BOS", "EOS"],
        )
    tokenizer = miditok.REMI(config)
    
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
    
    # process both train and val splits
    for split in ["train", "val"]:
        print(f"\n=== Processing {split.upper()} split ===")
        quantized_dir = Path(f"data/quantized/{split}")
    
        if not quantized_dir.exists():
            print(f"Error: {quantized_dir} does not exist!")
            print("Run quantize_all.py first to create quantized .pkl files")
            continue
    
        dataset = MIREXCustomDataset(
            quantized_dir=quantized_dir,
            tokenizer=tokenizer,
            max_seq_len=1024
        )
    
        print(f"{split.upper()} dataset size: {len(dataset)}")
    
        # quick test sample
        print("\nTesting sample loading:")
        for i in range(min(2, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: input_ids={sample['input_ids'].shape}, labels={sample['labels'].shape}")
    
        # quick DataLoader test
        print("\nTesting DataLoader:")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

    print("\nBoth train and val datasets loaded successfully!")

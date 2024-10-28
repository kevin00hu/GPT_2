import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tiktoken
import torch

from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

MAX_LENGTH = 1024

enc = tiktoken.get_encoding("r50k_base")

def encoding(examples):
    return {'input_ids': enc.encode_batch(examples['markdown'], num_threads=16, allowed_special={'<|endoftext|>'})}

def padding(tensor):
    return F.pad(tensor, (0, MAX_LENGTH - len(tensor)), mode='constant', value=enc.n_vocab-1)
    
def segment(examples):
    inputs = []
    labels = []

    for example in examples['input_ids']:
        for i in range(1, len(example), MAX_LENGTH):
            inputs.append( padding(torch.tensor(example[i-1:i+MAX_LENGTH-1])) )
            labels.append( padding(torch.tensor(example[i:i+MAX_LENGTH])) )

    return {
        "input": inputs,
        "label": labels
    }

def construct_dataset(dataset):
    ds = dataset.map(encoding, batched=True, num_proc=32, remove_columns=dataset.column_names) # tokenize
    ds = ds.map(segment, batched=True, num_proc=32, remove_columns=ds.column_names)
    return ds

def custom_collate_fn(batch):
    batch_input = [torch.tensor(item['input']) for item in batch]
    batch_label = [torch.tensor(item['label']) for item in batch]
    batch_input_tensor = torch.stack(batch_input) # (B, T)
    batch_label_tensor = torch.stack(batch_label) # (B, T)

    return batch_input_tensor, batch_label_tensor

def construct_dataloader(batch_size:int=8):
    dataset = load_dataset("neuralwork/arxiver", split='train')
    ds = construct_dataset(dataset)

    torch.manual_seed(42)
    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size   
    train_dataset, test_dataset = random_split(ds, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    dataloader = construct_dataloader()

    for input, label in dataloader:
        print(f"{input.shape=}")
        print(f"{label.shape=}")
        break

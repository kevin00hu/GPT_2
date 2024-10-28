from pathlib import Path

import torch
import torch.nn as nn
import torch.utils

from loguru import logger

from .model import GPT2
from data.dataset import construct_dataloader

logger.add("./log/train_log.txt", level="INFO", mode='w')

EPOCH = 1
BATCH_SIZE = 4
LR = 6e-5
SAVE_PATH = "./checkpoints/"
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

def train(epoch:int=EPOCH, batch_size:int=BATCH_SIZE, lr:float=LR):

    accumulation_steps = 64

    logger.info("Initialize model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2(50257)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        batch_size *= torch.cuda.device_count()
        logger.info(f"Multi-GPU detected, using DP and update batch_size to {batch_size}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Initialize dataloader")
    train_dataloader, test_dataloader = construct_dataloader(batch_size)
    total_batch = len(train_dataloader)

    logger.info("Start training!")
    for n_epoch in range(1, epoch+1):
        for batch_index, (inputs, labels) in enumerate(train_dataloader):
            batch_index += 1
            inputs = inputs.to(device) # (B, T)
            labels = labels.to(device) 

            outputs = model(inputs) # (B, T, V)
            B, T, V = outputs.shape
            loss = criterion(outputs.view(B*T, V), labels.view(B*T))

            loss.backward()

            if (batch_index) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                logger.info(f"[{(batch_index)/total_batch*100:.2f}%] {batch_index=} {loss=:.4f}")

            if batch_index % 1000 == 0:
                logger.info(f"[CHECKPOINT {batch_index}] Save the model to {SAVE_PATH + 'model.pth'}")
                torch.save(model, SAVE_PATH + 'model.pth')

    logger.info(f"Save the model to {SAVE_PATH + 'model.pth'}")
    torch.save(model, SAVE_PATH + "model.pth")
    
if __name__ == "__main__":
    train()
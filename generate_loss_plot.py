import re
import matplotlib.pyplot as plt

# read log file
LOG_PATH = "./log/train_log.txt"

with open(LOG_PATH, 'r') as f:
    content = f.readlines()

# extract loss and steps number 
batch_indices = []
loss_values = []
for line in content:
    match = re.search(r'batch_index=(\d+) loss=([\d.]+)', line)
    if match:
        batch_index = int(match.group(1))
        loss = float(match.group(2))
        batch_indices.append(batch_index)
        loss_values.append(loss)

# plot
plt.figure(figsize=(10, 6))
plt.plot(batch_indices, loss_values, label="Training Loss", color="blue")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.title("Training Loss Over Batches")
plt.legend()
plt.grid(True)
plt.savefig("./training_loss_plot.png")
plt.close()
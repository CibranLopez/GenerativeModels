import re
import matplotlib.pyplot as plt

# Define the path to your log file
log_file_path = "exps/train/single_sample/t_step_1/loss.log"

# Initialize lists to store each loss column
loss_columns = [[] for _ in range(4)]  # Assuming 4 loss values per line

# Regular expression to extract the loss values from the log
loss_pattern = re.compile(r"Era: \d+, ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)")

# Parse the log file
with open(log_file_path, "r") as log_file:
    for line in log_file:
        match = loss_pattern.search(line)
        if match:
            # Extract the four loss values and store them in respective lists
            for i, value in enumerate(match.groups()):
                loss_columns[i].append(float(value))

# Create a range for the losses
loss_range = range(len(loss_columns[0]))

# Plot the losses
plt.figure(figsize=(10, 6))
for i, loss in enumerate(loss_columns):
    if i == 1:
        break
    plt.plot(loss_range[::100], loss[::100], marker="o", linestyle="-", label=f"Loss {i + 1}")

plt.xlabel("Index")
plt.ylabel("Loss Value")
plt.title("Loss vs Index")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

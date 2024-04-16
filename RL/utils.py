from datasets import Dataset, Features, Array2D, Value, load_dataset, load_from_disk
import matplotlib.pyplot as plt
import numpy as np

def load_data(path):
    data = load_from_disk(path).with_format("torch")
    return data

def visualize_mazes(data, n = 20):
    print("======================")
    print(data.shape)
    print("---------------------")
    print(data[0])
    print("---------------------")
    print(data[0]['maze'])
    print("======================")

    rows = int(np.ceil(n / 4))  # Calculate the number of rows needed
    cols = 4  # Set the number of columns

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()  # Flatten the array to make iteration easier

    for i in range(n):
        axes[i].imshow(data[i]['maze'].numpy(), cmap='binary')
        axes[i].grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        axes[i].set_xticks(np.arange(-0.5, data[i]['maze'].shape[1], 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, data[i]['maze'].shape[0], 1), minor=True)
        # axes[i].axis('off')  # Optionally turn off the axis

    for i in range(n, len(axes)):
        axes[i].axis('off')  # Turn off the axis for unused subplots

    plt.tight_layout()
    plt.show()
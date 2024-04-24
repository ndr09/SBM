import matplotlib.pyplot as plt
import numpy as np

def plot_generations(title, means, bests):
    plt.title('Cartpole - Mulitple NNs compute deltaW')
    plt.plot(means, "r", label="mean")
    plt.plot(bests, "g", label="best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper left")
    plt.show()

def plot_weights(all_weights, nodes, generation, id, title):
    num_layers = len(nodes) - 1
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_layers + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))

    fig.suptitle(title)

    weights = []
    start_index = 0

    min = np.amin(all_weights[generation][id][99])
    max = np.amax(all_weights[generation][id][99])
    for i in range(num_layers):
        end_index = start_index + nodes[i] * nodes[i + 1]
        weights = np.array(all_weights[generation][id][99][start_index:end_index]).reshape(nodes[i], nodes[i + 1])

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]  # Select the appropriate subplot
        im = ax.imshow(weights, cmap='viridis', vmin=min, vmax=max)
        ax.set_title(f'Layer {i + 1} Weights')
        ax.set_xlabel(f'Layer {i + 1} Neurons')
        ax.set_ylabel(f'Layer {i} Neurons')
        ax.set_xticks(range(nodes[i + 1]))
        ax.set_yticks(range(nodes[i]))

        start_index = end_index

        # Add value to each cell
        for x in range(weights.shape[0]):
            for y in range(weights.shape[1]):
                if weights[x, y] > np.amax(weights)/2:
                  color = "black"
                else:
                  color = "white"
                ax.text(y, x, f'{weights[x, y]:.2f}', ha='center', va='center', color=color)


    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to leave space for the colorbar
    plt.subplots_adjust(right=0.85)  # Make room for the colorbar on the right side

    # Position the colorbar
    cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    # Adjust layout to prevent overlap
    plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_weights_gif(all_weights, nodes, generation, id, title):
    num_layers = len(nodes) - 1
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_layers + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))

    fig.suptitle(title)

    weights = []
    start_index = 0

    min = np.amin(all_weights[generation][id][:])
    print(f"min: {min}")
    max = np.amax(all_weights[generation][id][:])
    print(f"max: {max}")

    for i in range(num_layers):
        end_index = start_index + nodes[i] * nodes[i + 1]
        weights = np.array(all_weights[generation][id][0][start_index:end_index]).reshape(nodes[i], nodes[i + 1])

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]  # Select the appropriate subplot
        im = ax.imshow(weights, cmap='viridis', vmin = min, vmax=max)
        ax.set_title(f'Layer {i + 1} Weights')
        ax.set_xlabel(f'Layer {i + 1} Neurons')
        ax.set_ylabel(f'Layer {i} Neurons')
        ax.set_xticks(range(nodes[i + 1]))
        ax.set_yticks(range(nodes[i]))

        start_index = end_index

        # Add value to each cell
        for x in range(weights.shape[0]):
            for y in range(weights.shape[1]):
                if weights[x, y] > np.amax(weights)/2:
                  color = "black"
                else:
                  color = "white"
                ax.text(y, x, f'{weights[x, y]:.2f}', ha='center', va='center', color=color)

    def update(frame):
      weights = []
      start_index = 0
      for i in range(num_layers):
          end_index = start_index + nodes[i] * nodes[i + 1]
          weights = np.array(all_weights[generation][id][frame][start_index:end_index]).reshape(nodes[i], nodes[i + 1])

          row = i // num_cols
          col = i % num_cols
          ax = axes[row, col] if num_rows > 1 else axes[col]  # Select the appropriate subplot
          im = ax.imshow(weights, cmap='viridis', vmin=min, vmax=max)

          start_index = end_index

          # Clear previous text annotations
          for text in ax.texts:
            text.set_text("")


          # Add value to each cell
          for x in range(weights.shape[0]):
              for y in range(weights.shape[1]):
                  if weights[x, y] > np.amax(weights) / 2:
                      color = "black"
                  else:
                      color = "white"
                  ax.text(y, x, f'{weights[x, y]:.2f}', ha='center', va='center', color=color)

      return im,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, blit=True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to leave space for the colorbar
    plt.subplots_adjust(right=0.85)  # Make room for the colorbar on the right side

    # Position the colorbar
    cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.close()
    return ani

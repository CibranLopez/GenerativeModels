import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(csv_file, column_labels):
    """
    Plot specified columns from a CSV file with epochs on the x-axis and Loss on the y-axis.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the data.   
    column_labels : list
        List of column labels to plot.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Ensure all specified columns exist in the data
    for label in column_labels:
        if label not in data.columns:
            raise ValueError(f"Column '{label}' not found in the CSV file.")

    # Plot each column
    for label in column_labels:
        plt.plot(data.index, data[label], label=label)

    # Set axis labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Example usage:
# plot_columns('data.csv', ['loss1', 'loss2', 'loss3'])

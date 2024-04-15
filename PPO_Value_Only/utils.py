import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_critic_loss(file_path, figure_file):
    # Load losses from a file
    with open(file_path, 'r') as file:
        losses = file.readlines()
        losses = [float(loss.strip()) for loss in losses]
    
    # Calculate running average
    running_avg = np.zeros(len(losses))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(losses[max(0, i-100):(i+1)])
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(running_avg)
    plt.title('Running Average of Previous 100 Critic Losses')
    plt.xlabel('Batch Number')
    plt.ylabel('Critic Loss')
    plt.savefig(figure_file)
    plt.show()
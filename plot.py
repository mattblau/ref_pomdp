import numpy as np
import matplotlib.pyplot as plt

# def plot_critic_loss(file_path, figure_file):
#     # Load losses from a file
#     with open(file_path, 'r') as file:
#         losses = file.readlines()
#         losses = [float(loss.strip()) for loss in losses]
    
#     # Calculate running average
#     running_avg = np.zeros(len(losses))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(losses[max(0, i-100):(i+1)])
    
#     # Create the plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(running_avg)
#     plt.title('Running Average of Previous 100 Critic Losses')
#     plt.xlabel('Batch Number')
#     plt.ylabel('Critic Loss')
#     plt.savefig(figure_file)
#     plt.show()

# plot_critic_loss('critic_losses.txt', 'critic_loss_plot.png')

def plot_critic_loss(file_path, figure_file):
    # Load losses from a file
    with open(file_path, 'r') as file:
        losses = file.readlines()
        losses = [float(loss.strip()) for loss in losses]
    
    # Calculate running average
    window_size = 100
    running_avg = np.zeros(len(losses))
    for i in range(len(losses)):
        start_index = max(0, i - window_size + 1)
        running_avg[i] = np.mean(losses[start_index:(i+1)])
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(running_avg)
    plt.title('Running Average of Previous 100 Critic Losses')
    plt.xlabel('Batch Number')
    plt.ylabel('Critic Loss')
    plt.ylim(0, 30)  # Set y-axis limit to 30
    plt.savefig(figure_file)
    plt.show()

plot_critic_loss('critic_losses.txt', 'critic_loss_plot.png')
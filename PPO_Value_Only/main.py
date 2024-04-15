import gymnasium as gym
import numpy as np
import torch as T
from ppo_torch import LearningAgent
from utils import plot_learning_curve, plot_critic_loss

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    print(env.action_space.n)
    print(env.observation_space.shape)

    agent = LearningAgent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()[0]
        print("Observation: ", observation)
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            observation_tensor = T.tensor(observation, dtype=T.float).to(agent.critic.device)  
            # Now pass the tensor to the critic
            val = agent.critic(observation_tensor)
            val= T.squeeze(val).item()
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated
            n_steps += 1
            score += reward

            print("Observation: ", observation.dtype)
            print("Action: ", action.dtype)
            print("Val: ", type(val))
            print("Reward: ",type(reward))
            print("Done: ", type(done))
            agent.remember(observation, action, val, reward, done)

        
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    plot_critic_loss('critic_losses.txt', 'critic_loss_plot.png')
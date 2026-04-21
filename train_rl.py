import os
import torch
import numpy as np
from orbital_env import OrbitalEnvironment
from belief_estimator import BeliefEstimator
from rl_agent import PPO
from trajectory_predictor import TrajectoryPredictor
from trajectory_optimizer import SCPOptimizer

def train():
    ####### Hyperparameters #######
    env_name = "SpacecraftCollisionAvoidance"
    has_continuous_action_space = True
    max_ep_len = 200                    # max timesteps in one episode
    max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4        # print avg reward in the interval
    log_freq = max_ep_len * 2           # log avg reward in the interval
    save_model_freq = int(2e4)          # save model frequency
    action_std = 0.5                    # starting std for action distribution (local vars)
    action_std_decay_rate = 0.05        # linearly decay action_std (of latent)
    min_action_std = 0.1                # minimum action_std (of latent)
    action_std_decay_freq = int(2.5e4)  # action_std decay frequency

    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 40               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor network
    lr_critic = 0.001           # learning rate for critic network

    random_seed = 0             # set random seed if required
    ################################

    print("training environment name : " + env_name)

    env = OrbitalEnvironment()
    estimator = BeliefEstimator()

    # state space dimension
    state_dim = 6 # ROE state
    # action space dimension
    action_dim = 3 # [dv_r, dv_t, dv_n]

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # Initialize Feature 6.4 & 6.5 components
    predictor = TrajectoryPredictor(state_dim=state_dim, action_dim=action_dim, horizon=5)
    scp_optimizer = SCPOptimizer(action_dim=action_dim, safe_distance=0.01)

    # track total training time
    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        estimator.reset()
        current_ep_reward = 0

        # Initial belief from first observation (reset returns true state, but step returns observation)
        # For simplicity, we'll start with the reset state as the first belief
        belief_state = state 

        for t in range(1, max_ep_len + 1):

            # select action with policy
            raw_action = ppo_agent.select_action(belief_state)
            
            # Feature 6.4: Predict candidate avoidance trajectories
            predicted_trajectory = predictor.predict(belief_state, raw_action)
            
            # Feature 6.5: Refine initial action using SCP to ensure collision safety and fuel optimality
            optimized_action = scp_optimizer.optimize(belief_state, raw_action, predicted_trajectory)
            
            observation, true_state, reward, done, _ = env.step(optimized_action)
            
            # Feature 6.4: Online training step for trajectory predictor
            predictor.train_step(belief_state, optimized_action, true_state)

            # Update belief state using noisy observation
            belief_state = estimator.estimate(observation)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; decay action standard deviation
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log_f.write('{},{},{}\n'.format(i_episode, time_step, current_ep_reward))
                pass

            # print average reward
            if time_step % print_freq == 0:
                print(f"Episode : {i_episode} \t Timestep : {time_step} \t Reward : {current_ep_reward}")

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + f"PPO_{env_name}.pth")
                ppo_agent.save(f"PPO_{env_name}.pth")
                print("model saved")
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        i_episode += 1

    print("Training finished.")

if __name__ == '__main__':
    train()

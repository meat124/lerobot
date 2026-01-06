import gymnasium as gym
import gym_pusht
import matplotlib.pyplot as plt
import numpy as np
import time

def visualize_pusht_env():
    # 'Pusht-v0' is the standard ID for this environment
    # Using obs_type='pixels' for visual rendering
    try:
        env = gym.make('gym_pusht/PushT-v0', obs_type='pixels', render_mode='human')
        obs, info = env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        # Reset the environment to get initial observation
        observation, info = env.reset()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'env' in locals() and env is not None:
            env.close()

if __name__ == '__main__':
    visualize_pusht_env()

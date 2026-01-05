import gymnasium as gym
import gym_pusht
import matplotlib.pyplot as plt
import numpy as np

def visualize_pusht_env():
    # 'Pusht-v0' is the standard ID for this environment
    # Using obs_type='pixels' for visual rendering
    try:
        env = gym.make('gym_pusht/PushT-v0', obs_type='pixels', render_mode='rgb_array')
        
        # Reset the environment to get initial observation
        observation, info = env.reset()

        # Render the environment
        img = env.render()
        
        if img is not None:
            plt.imshow(img)
            plt.title("PushT-v0 Environment Visualization")
            plt.axis('off')
            plt.show()
        else:
            print("Environment rendering returned None. Make sure render_mode is set correctly.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'env' in locals() and env is not None:
            env.close()

if __name__ == '__main__':
    visualize_pusht_env()

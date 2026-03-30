import os
import sys

sys.path.append("coverage-gridworld")

import gymnasium as gym
import coverage_gridworld  
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = gym.make(
        "sneaky_enemies",
        predefined_map_list=None,
        activate_game_status=True,
    )
    env = Monitor(env)
    return env

def make_just_go_env():
    env = gym.make(
        "just_go",
        predefined_map_list=None,
        activate_game_status=False,
    )
    env = Monitor(env)
    return env


def make_sneaky_env(predefined_maps=None):
    env = gym.make(
        "sneaky_enemies",
        predefined_map_list=predefined_maps,
        activate_game_status=False,
    )
    env = Monitor(env)
    return env

custom_maps = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 4, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
]

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_just_go_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("models/ppo_stage1_just_go")
    env.close()

    # Stage 2: custom maps
    env = make_sneaky_env(predefined_maps=custom_maps)
    model.set_env(env)

    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("models/ppo_stage2_custom")
    env.close()

    # Stage 3: full sneaky_enemies
    env = make_sneaky_env()
    model.set_env(env)

    model.learn(total_timesteps=300000, progress_bar=True)
    model.save("models/ppo_stage3_sneaky")
    env.close()

    print("Training finished and model saved.")



if __name__ == "__main__":
    main()

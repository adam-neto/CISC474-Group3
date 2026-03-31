import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"

if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))


import coverage_gridworld
from coverage_gridworld import custom
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


custom.ACTIVE_OBSERVATION_SPACE = 1
custom.ACTIVE_OBSERVATION = 1
custom.ACTIVE_REWARD = 1


def train():
    env_id = "sneaky_enemies"
    
    n_envs = 4 
    
    train_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs={
            "render_mode": None,
            "predefined_map_list": None,
        }
    )

    # PPO Optimized for CPU & Local Observation
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log="./logs/"
    )

    print("Training the agent on CPU.")
    model.learn(total_timesteps=20_000)
    
    model.save("best_agent_v1")
    print("Completed. Saved as 'best_agent_v1.zip'")

if __name__ == "__main__":
    train()


# RUN WITH 
# python scripts/eval_sb3.py best_agent_v1.zip --observation-version 1 --reward-version 1 --render --episodes 50 --map-index 2
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


custom.ACTIVE_OBSERVATION_SPACE = 2
custom.ACTIVE_OBSERVATION = 2
custom.ACTIVE_REWARD = 2

AGENT_NAME = "best_agent_v1"
TOTAL_TIME_STEPS = 20_000


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

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-3, 
        ent_coef=0.1, 
        n_steps=1024, 
        batch_size=128,
        policy_kwargs={"net_arch": dict(pi=[256, 256], qf=[256, 256])},
        tensorboard_log="./logs/"
    )

    print(f"Training '{AGENT_NAME}'...")
    model.learn(total_timesteps=TOTAL_TIME_STEPS) 
    
    model.save(AGENT_NAME)
    print(f"Completed. Agent saved as '{AGENT_NAME}'.zip.")

if __name__ == "__main__":
    train()


# RUN WITH 
# python scripts/eval_sb3.py best_agent_v1.zip --observation-version 1 --reward-version 1 --render --episodes 50
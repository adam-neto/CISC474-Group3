import time
import gymnasium as gym
import coverage_gridworld
from stable_baselines3 import PPO

env = gym.make(
    "sneaky_enemies",
    render_mode="human",
    predefined_map_list=None,
    activate_game_status=False,
)

model = PPO.load("models/ppo_stage3_sneaky")

obs, info = env.reset()
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    print("action:", action)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.15)

print("Episode reward:", total_reward)
env.close()
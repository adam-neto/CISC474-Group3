import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# MODIFY THIS FILE
# use observation_space2, observation2, reward2 etc...


# Choose which versions are active
ACTIVE_OBSERVATION_SPACE = 1
ACTIVE_OBSERVATION = 1
ACTIVE_REWARD = 1

# Cache real environment settingss
CURRENT_GRID_SIZE = None
CURRENT_ENEMY_FOV_DISTANCE = None
DEFAULT_GRID_SIZE = 10
DEFAULT_ENEMY_FOV_DISTANCE = 4


def _get_env_constants():
    grid_size = CURRENT_GRID_SIZE if CURRENT_GRID_SIZE is not None else DEFAULT_GRID_SIZE
    enemy_fov_distance = (
        CURRENT_ENEMY_FOV_DISTANCE
        if CURRENT_ENEMY_FOV_DISTANCE is not None
        else DEFAULT_ENEMY_FOV_DISTANCE
    )
    return grid_size, enemy_fov_distance


def observation_space(env: gym.Env) -> gym.spaces.Space:
    global CURRENT_GRID_SIZE, CURRENT_ENEMY_FOV_DISTANCE

    CURRENT_GRID_SIZE = env.grid_size
    CURRENT_ENEMY_FOV_DISTANCE = getattr(
        env,
        "enemy_fov_distance",
        DEFAULT_ENEMY_FOV_DISTANCE,
    )
    
    if ACTIVE_OBSERVATION_SPACE == 0:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=env.grid.flatten().shape,
            dtype=np.uint8,
        )
    elif ACTIVE_OBSERVATION_SPACE == 1:
        return observation_space1(env)
    else:
        raise ValueError(f"Unknown observation space version: {ACTIVE_OBSERVATION_SPACE}")


def observation(grid: np.ndarray):
    if ACTIVE_OBSERVATION == 0:
        return grid.flatten()
    elif ACTIVE_OBSERVATION == 1:
        return observation1(grid)
    else:
        raise ValueError(f"Unknown observation version: {ACTIVE_OBSERVATION}")


def reward(info: dict) -> float:
    if ACTIVE_REWARD == 0:
        return 0
    elif ACTIVE_REWARD == 1:
        return reward1(info)
    else:
        raise ValueError(f"Unknown reward version: {ACTIVE_REWARD}")


# JEREMY
def observation_space1(env: gym.Env) -> gym.spaces.Space:
    grid_size = env.grid_size
    obs_size = grid_size * grid_size

    return gym.spaces.Box(
        low=0,
        high=5,
        shape=(obs_size,),
        dtype=np.float32,
    )


def observation1(grid: np.ndarray):
    grid_size = grid.shape[0]

    simplified_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = grid[i, j]

            if (cell == [0, 0, 0]).all():
                simplified_grid[i, j] = 0
            elif (cell == [255, 255, 255]).all():
                simplified_grid[i, j] = 1
            elif (cell == [101, 67, 33]).all():
                simplified_grid[i, j] = 2
            elif (cell == [255, 0, 0]).all():
                simplified_grid[i, j] = 3
            elif (cell == [255, 127, 127]).all():
                simplified_grid[i, j] = 4
            else:
                simplified_grid[i, j] = 5

    return simplified_grid.flatten()


def reward1(info: dict) -> float:
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    cells_remaining = info["cells_remaining"]

    grid_size, enemy_fov_distance = _get_env_constants()
    agent_cell = divmod(agent_pos, grid_size)

    reward = 0.0

    if game_over:
        return -100.0

    if cells_remaining == 0:
        return 100.0

    if new_cell_covered:
        reward += 5.0
    else:
        reward -= 0.5

    for enemy in enemies:
        next_orientation = (enemy.orientation + 1) % 4

        for i in range(1, enemy_fov_distance + 1):
            if next_orientation == 0:
                fy, fx = enemy.y, enemy.x - i
            elif next_orientation == 1:
                fy, fx = enemy.y + i, enemy.x
            elif next_orientation == 2:
                fy, fx = enemy.y, enemy.x + i
            else:
                fy, fx = enemy.y - i, enemy.x

            if fy < 0 or fx < 0 or fy >= grid_size or fx >= grid_size:
                break

            if agent_cell == (fy, fx):
                reward -= 10.0
                break

    reward -= 0.1
    return reward
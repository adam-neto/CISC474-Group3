import numpy as np
import gymnasium as gym
from dataclasses import dataclass

# Choose which versions are active
ACTIVE_OBSERVATION_SPACE = 1
ACTIVE_OBSERVATION = 1
ACTIVE_REWARD = 1

# Default environement variables
DEFAULT_GRID_SIZE = 10
DEFAULT_ENEMY_FOV_DISTANCE = 4


# Helper functions
@dataclass(frozen=True)
class EnvConstants:
    grid_size: int
    enemy_fov_distance: int


def get_env_constants(env: gym.Env | None = None, info: dict | None = None) -> EnvConstants:
    """
    Resolve environment constants from the environment or info dict.

    Priority:
    1. env attributes
    2. info dictionary values
    3. module defaults
    """
    if env is not None:
        return EnvConstants(
            grid_size=getattr(env, "grid_size", DEFAULT_GRID_SIZE),
            enemy_fov_distance=getattr(
                env,
                "enemy_fov_distance",
                DEFAULT_ENEMY_FOV_DISTANCE,
            ),
        )

    if info is not None:
        return EnvConstants(
            grid_size=info.get("grid_size", DEFAULT_GRID_SIZE),
            enemy_fov_distance=info.get(
                "enemy_fov_distance",
                DEFAULT_ENEMY_FOV_DISTANCE,
            ),
        )

    return EnvConstants(
        grid_size=DEFAULT_GRID_SIZE,
        enemy_fov_distance=DEFAULT_ENEMY_FOV_DISTANCE,
    )

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
    elif ACTIVE_OBSERVATION_SPACE == 2:
        return observation_space2(env)
    elif ACTIVE_OBSERVATION_SPACE == 3:
        return observation_space3(env)

    else:
        raise ValueError(f"Unknown observation space version: {ACTIVE_OBSERVATION_SPACE}")


def observation(grid: np.ndarray):
    if ACTIVE_OBSERVATION == 0:
        return grid.flatten()
    elif ACTIVE_OBSERVATION == 1:
        return observation1(grid)
    elif ACTIVE_OBSERVATION == 2:
        return observation2(grid)
    elif ACTIVE_OBSERVATION == 3:
        return observation3(grid)
    else:
        raise ValueError(f"Unknown observation version: {ACTIVE_OBSERVATION}")


def reward(info: dict) -> float:
    if ACTIVE_REWARD == 0:
        return 0
    elif ACTIVE_REWARD == 1:
        return reward1(info)
    elif ACTIVE_REWARD == 2:
        return reward2(info)
    elif ACTIVE_REWARD == 3:
        return reward3(info)
    else:
        raise ValueError(f"Unknown reward version: {ACTIVE_REWARD}")



# JEREMY

def observation_space1(env: gym.Env) -> gym.spaces.Space:
    return gym.spaces.Box(
        low=0,
        high=5,
        shape=(env.grid_size * env.grid_size,),
        dtype=np.float32,
    )


def observation1(grid: np.ndarray):
    color_map = {
        (0, 0, 0): 0,          # empty / uncovered
        (255, 255, 255): 1,    # covered
        (101, 67, 33): 2,      # wall
        (255, 0, 0): 3,        # enemy FOV / danger
        (255, 127, 127): 4,    # lighter danger
    }

    grid_size = grid.shape[0]
    simplified_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            simplified_grid[i, j] = color_map.get(tuple(grid[i, j]), 5)

    return simplified_grid.flatten()


from collections import deque

def reward1(info: dict) -> float:
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    prev_agent_pos = info.get("prev_agent_pos", agent_pos)
    prev_prev_agent_pos = info.get("prev_prev_agent_pos", prev_agent_pos)

    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    cells_remaining = info["cells_remaining"]
    visited_matrix = info.get("visited_matrix")
    stagnation_steps = info.get("stagnation_steps", 0)

    constants = get_env_constants(info=info)
    grid_size = constants.grid_size
    enemy_fov_distance = constants.enemy_fov_distance

    agent_cell = divmod(agent_pos, grid_size)
    prev_agent_cell = divmod(prev_agent_pos, grid_size)
    prev_prev_agent_cell = divmod(prev_prev_agent_pos, grid_size)

    if game_over:
        return -30.0
    if cells_remaining == 0:
        return 100.0

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def in_bounds(y, x):
        return 0 <= y < grid_size and 0 <= x < grid_size

    def get_fov_cells():
        fov = set()
        for enemy in enemies:
            idx = enemy.orientation
            dy, dx = directions[idx]

            for step in range(1, enemy_fov_distance + 1):
                fy = enemy.y + dy * step
                fx = enemy.x + dx * step
                if not in_bounds(fy, fx):
                    break
                fov.add((fy, fx))
        return fov

    def nearest_unvisited_distance(start):
        if visited_matrix is None:
            return 0

        sy, sx = start
        q = deque([(sy, sx, 0)])
        seen = {(sy, sx)}

        while q:
            y, x, d = q.popleft()

            if visited_matrix[y][x] == 0:
                return d

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if in_bounds(ny, nx) and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    q.append((ny, nx, d + 1))

        return 0

    reward = 0.0
    fov_cells = get_fov_cells()

    moved = agent_pos != prev_agent_pos
    curr_in_fov = agent_cell in fov_cells
    prev_in_fov = prev_agent_cell in fov_cells

    curr_dist = nearest_unvisited_distance(agent_cell)
    prev_dist = nearest_unvisited_distance(prev_agent_cell)

    if new_cell_covered:
        reward += 10.0

    dist_improvement = prev_dist - curr_dist
    if dist_improvement > 0:
        reward += 1.5 * dist_improvement

    if not moved:
        reward -= 2.0 + 0.5 * stagnation_steps

    if agent_pos == prev_prev_agent_pos and agent_pos != prev_agent_pos:
        reward -= 3.0

    if visited_matrix is not None:
        times_visited = visited_matrix[agent_cell]
        reward -= min(2.5, 0.35 * times_visited)

    if curr_in_fov:
        reward -= 2.5
    if prev_in_fov and not curr_in_fov:
        reward += 1.5

    reward -= 0.05

    return reward



# MONICA

def observation_space2(env: gym.Env) -> gym.spaces.Space:
    # 5x5 local view centered on agent
    return gym.spaces.Box(
        low=0,
        high=5,
        shape=(25,),
        dtype=np.float32,
    )


def observation2(grid: np.ndarray):
    grid_size = grid.shape[0]

    # find agent position (grey cell)
    agent_pos = None
    for i in range(grid_size):
        for j in range(grid_size):
            if (grid[i, j] == [160, 161, 161]).all():
                agent_pos = (i, j)
                break
        if agent_pos:
            break

    # fallback (should not happen)
    if agent_pos is None:
        return np.zeros(25, dtype=np.float32)

    ay, ax = agent_pos

    # build 5x5 local window
    local = np.zeros((5, 5), dtype=np.float32)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y = ay + dy
            x = ax + dx

            if 0 <= y < grid_size and 0 <= x < grid_size:
                cell = grid[y, x]

                if (cell == [0, 0, 0]).all():
                    local[dy + 2, dx + 2] = 0
                elif (cell == [255, 255, 255]).all():
                    local[dy + 2, dx + 2] = 1
                elif (cell == [101, 67, 33]).all():
                    local[dy + 2, dx + 2] = 2
                elif (cell == [255, 0, 0]).all():
                    local[dy + 2, dx + 2] = 3
                elif (cell == [255, 127, 127]).all():
                    local[dy + 2, dx + 2] = 4
                else:
                    local[dy + 2, dx + 2] = 5
            else:
                local[dy + 2, dx + 2] = 2  # treat out-of-bounds as wall

    return local.flatten()


def reward2(info: dict) -> float:
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    cells_remaining = info["cells_remaining"]

    constants = get_env_constants(info=info)
    grid_size = constants.grid_size
    agent_cell = divmod(agent_pos, grid_size)

    reward = 0.0

    # death
    if game_over:
        return -100.0

    # completion
    if cells_remaining == 0:
        return 150.0

    # exploration
    if new_cell_covered:
        reward += 5.0
    else:
        reward -= 0.2  # penalize revisits
        
    # proximity to enemies (Manhattan distance)
    for enemy in enemies:
        dist = abs(agent_cell[0] - enemy.y) + abs(agent_cell[1] - enemy.x)

        if dist <= 2:
            reward -= 0.3  # danger zone
        else:
            reward += 0.02  # safe movement bonus

    # time penalty
    reward -= 0.1

    return reward



# ENQI

def observation_space3(env: gym.Env) -> gym.spaces.Space:
    grid_size = env.grid_size
    obs_size = grid_size * grid_size

    return gym.spaces.Box(
        low=0,
        high=6,
        shape=(obs_size,),
        dtype=np.float32,
    )


def observation3(grid: np.ndarray):
    grid_size = grid.shape[0]
    simplified_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = grid[i, j]

            if (cell == [0, 0, 0]).all():              # black
                simplified_grid[i, j] = 0
            elif (cell == [255, 255, 255]).all():      # white
                simplified_grid[i, j] = 1
            elif (cell == [101, 67, 33]).all():        # wall
                simplified_grid[i, j] = 2
            elif (cell == [255, 0, 0]).all():          # red
                simplified_grid[i, j] = 3
            elif (cell == [255, 127, 127]).all():      # light red
                simplified_grid[i, j] = 4
            elif (cell == [160, 161, 161]).all():      # agent grey
                simplified_grid[i, j] = 5
            elif (cell == [31, 198, 0]).all():         # enemy green
                simplified_grid[i, j] = 6

    return simplified_grid.flatten()


def reward3(info: dict) -> float:
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    cells_remaining = info["cells_remaining"]

    constants = get_env_constants(info=info)
    grid_size = constants.grid_size
    enemy_fov_distance = constants.enemy_fov_distance
    agent_cell = divmod(agent_pos, grid_size)

    reward = 0.0

    if game_over:
        return -100.0

    if cells_remaining == 0:
        return 200.0

    if new_cell_covered:
        reward += 5.0
    else:
        reward -= 1.0

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
                reward -= 0.5
                break

    reward -= 0.01
    return reward

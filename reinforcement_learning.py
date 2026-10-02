import numpy as np
import random

from sklearn.linear_model import SGDRegressor

ROWS = 10
COLS = 10

START = (0,0)
TARGET = (9,9)

ACTIONS ={
    0:"Up",
    1:"Down",
    2:"Left",
    3:"Right",
}

REWARD_NORMAL = -1
REWARD_INVALID = -5
REWARD_WALL = -8
REWARD_DANGER = -15
REWARD_TARGET = 100

GRID = [
    list("Aoo#oooDoo"),
    list("o#oooo#ooo"),
    list("ooo#Doooo#"),
    list("#oooo#Dooo"),
    list("oo#oooo#Do"),
    list("Dooo#oooo#"),
    list("o#Doooo#oo"),
    list("oooo#Doooo"),
    list("#Doo#ooDoo"),
    list("oo#Doo#ooT"),
]

def get_cell_type(position):
    row, col = position
    return GRID[row][col]

def take_action(state, action):
    row, col = state

    new_row = row
    new_col = col

    if action == 0:
        new_row -= 1

    elif action == 1:
        new_row += 1

    elif action == 2:
        new_col -= 1

    elif action == 3:
        new_col += 1

    if (
        new_row < 0
        or new_row >= ROWS
        or new_col < 0
        or new_col >= COLS
    ): return state, REWARD_INVALID, False, "Invalid"

    cell = GRID[new_row][new_col]

    if cell == "#":
        return state, REWARD_WALL, False, "Wall"

    next_state = (new_row, new_col)

    if cell == "D":
        return next_state, REWARD_DANGER, False, "Danger"

    if cell == "T":
        return next_state, REWARD_TARGET, True, "Target"

    return next_state, REWARD_NORMAL, False, "Path"

def state_action_features(state, action):
    row, col = state

    return np.array([
        row / (ROWS - 1),
        col / (COLS - 1),
        action / 3
    ]).reshape(1, -1)

models = []

def initialize_models():
    global models

    models = []

    for action in range(4):
        model = SGDRegressor(
            learning_rate="constant",
            eta0=0.01,
            random_state=42
        )

        X = state_action_features(START, action)
        y = np.array([0.0])

        model.partial_fit(X, y)

        models.append(model)
  
def get_q_values(state):
    q_values = []

    for action in range(4):
        X = state_action_features(state, action)
        q = models[action].predict(X)[0]
        q_values.append(q)

    return np.array(q_values)

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)

    q_values = get_q_values(state)

    return int(np.argmax(q_values))

EPISODES = 1000
MAX_STEPS = 200

GAMMA = 0.95

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995

def train_agent():

    initialize_models()

    epsilon = INITIAL_EPSILON
    successful_episodes = 0
    rewards_history = []

    for episode in range(EPISODES):

        state = START
        total_reward = 0

        for step in range(MAX_STEPS):

            action = choose_action(state, epsilon)

            next_state, reward, done, cell_type = take_action(state, action)

            if done:
                target_q = reward
            else:
                next_q_values = get_q_values(next_state)

                target_q = reward + GAMMA * np.max(next_q_values)

        models[action].partial_fit(
            state_action_features(state, action),
            np.array([target_q])
                    )
        total_reward += reward

        state = next_state

        if done:
            successful_episodes += 1
            break

    rewards_history.append(total_reward)


    epsilon = max(
                MIN_EPSILON,
                epsilon * EPSILON_DECAY
            )

    return {

            "episodes": EPISODES,

            "successful": successful_episodes,

            "average_reward": np.mean(rewards_history),

            "final_epsilon": epsilon

        }

def validate_grid():
    counts = {
        "A": 0,
        "T": 0,
        "o": 0,
        "#": 0,
        "D": 0
    }

    for row in GRID:
        for cell in row:
            if cell in counts:
                counts[cell] += 1

    print("Environment validation")
    print("----------------------")
    print("Agent:", counts["A"])
    print("Target:", counts["T"])
    print("Path:", counts["o"])
    print("Walls:", counts["#"])
    print("Danger:", counts["D"])
    print("Total:", sum(counts.values()))

if __name__ == "__main__":
    validate_grid()
    initialize_models()
    print("\nQ values iniciales:")
    print(get_q_values((0,0)))
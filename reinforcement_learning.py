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
    list("oo###Doooo"),
    list("#Doo#DooDo"),
    list("oo#ooo#ooT"),
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

def get_valid_actions(state):
    valid_actions = []

    for action in range(4):
        next_state, reward, done, cell = take_action(state,action)
        if cell not in ["Invalid","Wall"]:

            valid_actions.append(action)

    return valid_actions

models = []

def state_features(state):
    row, col = state
    x = np.zeros((1, ROWS * COLS))
    x[0, row * COLS + col] = 1.0
    return x

def initialize_models():
    global models

    models = []

    for action in range(4):
        model = SGDRegressor(
            learning_rate="constant",
            eta0=0.1,
            alpha=0.0,
            fit_intercept=False,
            random_state=42
        )

        X_init = np.zeros((1, ROWS * COLS))
        model.partial_fit(X_init, np.array([0.0]))

        models.append(model)
  
def get_q_values(state):
    X = state_features(state)
    return np.array([models[a].predict(X)[0] for a in range(4)])

def choose_action(state, epsilon):
    valid_actions = get_valid_actions(state)
    
    if random.random() < epsilon:
        return random.choice(valid_actions)

    q_values = get_q_values(state)
    valid_q = {}

    for action in valid_actions:
        valid_q[action] = q_values[action]

    return max(valid_q,key=valid_q.get)

EPISODES = 5000
MAX_STEPS = 200

GAMMA = 0.95

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995

def train_agent():

    initialize_models()

    epsilon = INITIAL_EPSILON
    successful = 0
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

                state_features(state),
                np.array([target_q])
                )
            total_reward += reward

            state = next_state

            if done:
                successful += 1
                break

        rewards_history.append(total_reward)


        epsilon = max(
                MIN_EPSILON,
                epsilon * EPSILON_DECAY
            )
    average_reward = np.mean(rewards_history)

    print("\nTraining completed")
    print("--------------------")
    print("\nSuccessful episodes",successful)
    print("\nAverage reward",average_reward)
    print("\nFinal epsilon",epsilon)

    return {
             "episodes": EPISODES,
             "successful": successful,
             "average_reward": average_reward,
             "final_epsilon": epsilon
    }

def evaluate_agent():
    state = START
    path = [state]

    steps = []

    total_reward = 0

    for step in range(MAX_STEPS):

        action = choose_action(state,0)

        next_state, reward, done, cell_type = take_action(state,action)

        steps.append({
            "step": step + 1,
            "state": state,
            "action": ACTIONS[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward

        })

        path.append(next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    return {
        "goal_reached": state == TARGET,
        "movements": len(steps),
        "total_reward": total_reward,
        "path": path,
        "steps": steps
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
    print("\nTraining agent")
    results=train_agent()
    print("\nTrain results")
    print(results)
    print("\nQ values after training")
    print(get_q_values(START))
    print("\nEvaluation")
    evaluation = evaluate_agent()
    print(evaluation)
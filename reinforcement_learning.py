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
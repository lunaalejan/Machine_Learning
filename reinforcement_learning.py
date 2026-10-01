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
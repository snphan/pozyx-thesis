import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    '#1 grab something from freezer',
    'quiet standing',
    '#2 grab something from freezer',
    'quiet standing',
    '#3 grab something from freezer',
    'quiet standing',
    '#4 grab something from freezer',
    'quiet standing',
    '#5 grab something from freezer',
    'quiet standing',
    '10 seconds elapsed'
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'OPENFREEZER')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"OPENFREEZER_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
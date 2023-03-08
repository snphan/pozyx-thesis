import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    'put bread on plate',
    'shake and squeeze mayo',
    'put meat',
    'put cheese',
    'put tomato',
    'put salad',
    'put bread',
    'done sandwich',
    'quiet standing',
    '10 seconds elapsed',
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'ASSEMBLESANDWICH')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"ASSEMBLESANDWICH_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
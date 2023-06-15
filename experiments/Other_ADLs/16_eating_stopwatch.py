import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing ',
    'eat', #Front door
    'quiet standing ',
    'eat', #Kitchen
    'quiet standing ',
    'eat', #Living Room Table
    'quiet standing',
    '5 seconds elapsed',
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'EATING')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"EATING_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
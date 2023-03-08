import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    '#1 plate cabinet to counter',
    'quiet standing',
    '#1 plate counter to cabinet',
    'quiet standing',
    '#2 plate cabinet to counter',
    'quiet standing',
    '#2 plate counter to cabinet',
    'quiet standing',
    '#3 plate cabinet to counter',
    'quiet standing',
    '#3 plate counter to cabinet',
    'quiet standing',
    '#4 plate cabinet to counter',
    'quiet standing',
    '#4 plate counter to cabinet',
    'quiet standing',
    '#5 plate cabinet to counter',
    'quiet standing',
    '#5 plate counter to cabinet',
    'quiet standing',
    '10 seconds elapsed'
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'GETPLATE')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"GETPLATE_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
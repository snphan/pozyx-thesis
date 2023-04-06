import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    '#1 open dishwasher',
    'quiet standing',
    '#2 open dishwasher',
    'quiet standing',
    '#3 open dishwasher',
    'quiet standing',
    '#4 open dishwasher',
    'quiet standing',
    '#5 open dishwasher',
    'quiet standing',
    '5 seconds elapsed'
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'OPENDISHWASHER')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"OPENDISHWASHER_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
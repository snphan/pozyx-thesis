import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    '5 sec elapsed',
    '#1 brush teeth',
    'done',
    'quiet standing',
    '5 sec elapsed',
    '#2 brush teeth',
    'done',
    'quiet standing',
    '5 sec elapsed',
    '#3 brush teeth',
    'done',
    'quiet standing',
    '5 sec elapsed'
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'BRUSHTEETH')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"BRUSHTEETH_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
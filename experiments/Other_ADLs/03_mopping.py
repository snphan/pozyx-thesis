import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing (living entrance)', # 5 seconds
    'mop living room', # 10 seconds
    'move to bedroom',
    'quiet standing (bedroom door)', # 5 seconds
    'mop bedroom', # 10 seconds
    'move to bathroom',
    'quiet standing (bathroom door)', # 5 seconds
    'mop bathroom', # 10 seconds
    'move to kitchen',
    'quiet standing (kitchen)', # 5 seconds
    'mop kitchen', # 10 seconds
    'move back to living entrance',
    'quiet standing (living entrance)', # 5 seconds
    '5 seconds elapsed',
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'MOP')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"MOP_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
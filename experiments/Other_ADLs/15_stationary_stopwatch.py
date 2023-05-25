import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing ',
    'stand', #Front door
    'quiet standing ',
    'stand', #Kitchen
    'quiet standing ',
    'stand', #Living Room Table
    'quiet standing ',
    'stand', #Hallway
    'quiet standing ',
    'stand', #Bedroom Chair
    'quiet standing ',
    'stand', #Bedroom Closet
    'quiet standing ',
    'stand', #Bathroom Sink
    'quiet standing',
    '5 seconds elapsed',
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'STATIONARY')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"STATIONARY_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
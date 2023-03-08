import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing',
    '#1 open faucet',
    'wash tomato',
    'close faucet',
    'dry hands',
    'quiet standing',
    '#2 open faucet',
    'wash tomato',
    'close faucet',
    'dry hands',
    'quiet standing',
    '#3 open faucet',
    'wash tomato',
    'close faucet',
    'dry hands',
    'quiet standing',
    '#4 open faucet',
    'wash tomato',
    'close faucet',
    'dry hands',
    'quiet standing',
    '#5 open faucet',
    'wash tomato',
    'close faucet',
    'dry hands',
    'quiet standing',
    '10 seconds elapsed'

]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'WASHTOMATO')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"WASHTOMATO_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
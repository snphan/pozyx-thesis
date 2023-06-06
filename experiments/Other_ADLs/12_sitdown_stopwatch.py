import time
from pathlib import Path

TRANSITIONS = [
    'quiet standing (standing at toilet)', # 5 seconds
    'sitdown', 
    'quiet standing (standing at couch)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)', # 5 seconds
    'sitdown', 
    'quiet standing (standing at desk)',
    'sitdown',
    'quiet standing (standing at bedroom chair)',
    'sitdown',
    'quiet standing',
    'sitdown', 
    'quiet standing (standing at couch)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)', # 5 seconds
    'sitdown', 
    'quiet standing (standing at desk)',
    'sitdown',
    'quiet standing (standing at bedroom chair)',
    'sitdown', 
    'quiet standing (standing at couch)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)',
    'sitdown',
    'quiet standing (standing at kitchen table 2/2)', # 5 seconds
    'sitdown', 
    'quiet standing (standing at desk)',
    'sitdown',
    'quiet standing',
    '5 seconds elapsed',
]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.joinpath('data', 'SITDOWN')
    data_dir.mkdir(parents=True, exist_ok=True)
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    participant = input("Input Participant Initials: ")
    with open(data_dir.joinpath(f"SITDOWN_{participant}_A{anchor_num}_{trial}.txt"), "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
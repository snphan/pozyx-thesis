import time

TRANSITIONS = [
    'quiet standing',
    '#1 grab something from fridge',
    'quiet standing',
    '#2 grab something from fridge',
    'quiet standing',
    '#3 grab something from fridge',
    'quiet standing',
    '#4 grab something from fridge',
    'quiet standing',
    '#5 grab something from fridge',
    'quiet standing',
    'move around',
    'back to fridge',
    '10 seconds elapsed'
]

if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"OPENFRIDGE_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
import time

TRANSITIONS = [
    'quiet standing',
    'slice tomato',
    'quiet standing',
    'move around',
    'quiet standing',
    '10 seconds elapsed',
]

if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"SLICETOMATO_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
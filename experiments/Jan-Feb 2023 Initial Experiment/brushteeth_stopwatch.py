import time

TRANSITIONS = [
    'wait at kitchen sink',
    '10 sec elapsed',
    'kitchen sink brushing start',
    '20 sec elapsed',
    'wait at bathroom sink',
    '10 sec elapsed',
    'bathroom sink brushing start',
    '20 sec elapsed',
    'back to kitchen wait',
    '10 sec elapsed'
]

if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"BRUSHTEETH_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
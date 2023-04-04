import time

TRANSITIONS = [
    'Go Refrigerator',
    '10 sec elapsed',
    'Dishwasher',
    '10 sec elapsed',
    'Sink',
    '10 sec elapsed',
    'Counter',
    '10 sec elapsed',
    'Stove',
    '10 sec elapsed',
    'Microwave',
    '10 sec elapsed',
    'Refrigerator',
    '10 sec elapsed',
]

if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"KITCHEN_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
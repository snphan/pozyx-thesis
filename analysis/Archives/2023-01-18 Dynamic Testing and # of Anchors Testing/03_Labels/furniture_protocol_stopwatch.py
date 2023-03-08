import time

TRANSITIONS = [
    'Go',
    '10 sec elapsed',
    'Stop at dining table',
    '10 sec elapsed',
    'First chair',
    '10 sec elapsed',
    'Second chair',
    '10 sec elapsed',
    'Third chair',
    '10 sec elapsed',
    'Fourth chair',
    '10 sec elapsed',
    'Desk wait',
    '10 sec elapsed',
    'Desk chair',
    '10 sec elapsed',
    'Couch wait',
    '10 sec elapsed',
    'Sit on couch',
    '10 sec elapsed',
    'Bed wait',
    '10 sec elapsed',
    'On bed',
    '10 sec elapsed',
    'bed sofa wait',
    '10 sec elapsed',
    'bed sofa',
    '10 sec elapsed',
    'washroom sink wait',
    '10 sec elapsed',
    'washroom sink',
    '10 sec elapsed',
    'toilet wait',
    '10 sec elapsed',
    'toilet',
    '10 sec elapsed',
    'bath wait',
    '10 sec elapsed',
    'bath',
    '10 sec elapsed',
    'oven wait',
    '10 sec elapsed',
    'oven',
    '10 sec elapsed',
    'fridge wait',
    '10 sec elapsed',
    'fridge',
    '10 sec elapsed',
]
if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"FURNITURE_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
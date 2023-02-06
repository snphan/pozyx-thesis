import time

TRANSITIONS = [
    'Go Hallway between kitchen and living',
    '10 sec elapsed',
    'Sit',
    '10 sec elapsed',
    'Living Room',
    '10 sec elapsed',
    'sit',
    '10 sec elapsed',
    'bedroom',
    '10 sec elapsed',
    'sit',
    '10 sec elapsed',
    'hallway between bedroom and bathroom',
    '10 sec elapsed',
    'sit',
    '10 sec elapsed',
    'bathroom',
    '10 sec elapsed',
    'sit',
    '10 sec elapsed',
    'kitchen',
    '10 sec elapsed',
    'sit',
    '10 sec elapsed',
    'hallway between kitchen and living',
    '10 sec elapsed',
]

if __name__ == "__main__":
    start_time = None
    counter = 0
    trial = input("Trial Number: ")
    anchor_num = input("Number of Anchors: ")
    with open(f"ZPOS_A{anchor_num}_{trial}.txt", "w+") as f:
        while counter < len(TRANSITIONS):
            input("")
            f.write(f"{TRANSITIONS[counter]}: {time.time()}\n")
            print(f"{TRANSITIONS[counter]}: {time.time()}")
            counter += 1
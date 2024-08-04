import sys

file_name = sys.argv[1]
start_time = sys.argv[2]

with open(file_name, 'r') as f:
    data = [line.replace("\n", "") for line in f.readlines()]
    data = [line.split(": ") for line in data]

with open(file_name+"_NEWSTART", 'w') as f:
    for line in data:
        f.write(f"{line[0]}: {float(line[1]) + float(start_time)}\n")


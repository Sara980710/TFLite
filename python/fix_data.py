import os 

dir = "/home/sara/Documents/Master-thesis/TFLite/memory"
for file in os.listdir(dir):
    with open(f"{dir}/{file}") as f:
        lines = f.readlines()

    parts = []
    last = 0
    for i, l in enumerate(lines):
        if "VIRT" in l:
            parts.append(lines[last:i])
            last = i
    parts.append(lines[last:])
    parts = list(filter(None, parts))
    assert(len(parts) <= 2)
    if len(parts) == 2:
        with open(f"{dir}/yolo_{file}", 'w') as f:
            f.writelines(parts[0])
        with open(f"{dir}/class_{file}", 'w') as f:
            f.writelines(parts[1])
        os.remove(f"{dir}/{file}")
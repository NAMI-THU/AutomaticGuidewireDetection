import csv
import os

folder = "data/CLINIC/images"
tasks = ["val", "test", "train"]
instruments = ["AC", "GW", "PC", "SR"]
augmentation_names = ["hf","vf","rot90","colch","crop"]
num_sequences = 5
counts = {}

for instrument in instruments:
    for sequence in range(1,num_sequences+1):
        counts[f"{instrument}-p{sequence}"] = {"train-aug": 0, "val-aug": 0, "test-aug": 0, "bg-train": 0, "bg-val": 0, "bg-test": 0, "source-train":0, "source-val":0, "source-test":0}

for task in tasks:
    path = os.path.join(folder, task)
    for filename in os.listdir(path):
        for sequence in range(1,num_sequences+1):
            if f"p{sequence}" in filename:
                for instrument in instruments:
                    if instrument in filename:
                        any_augmentation = False
                        for augmentation_name in augmentation_names:
                            if augmentation_name in filename:
                                any_augmentation = True
                        if any_augmentation is True:
                            taskname = f"{task}-aug" if "BG" not in filename else f"bg-{task}"
                            counts[f"{instrument}-p{sequence}"][taskname] += 1
                        else:
                            counts[f"{instrument}-p{sequence}"][f"source-{task}"] += 1

for entry in counts:
    print(entry, counts[entry])

with open('counts.csv', 'w', newline='') as csvfile:
    fieldnames = ['instrument_sequence'] + list(counts[list(counts.keys())[0]].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry, count_dict in counts.items():
        row = {'instrument_sequence': entry}
        row.update(count_dict)
        writer.writerow(row)
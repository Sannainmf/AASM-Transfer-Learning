# peek_rans.py
# Just prints the first 20 lines of the first RANS training sample

import glob
import os

path = r"C:\Users\areen\Documents\trainingData-20260406T015436Z-1-001\trainingData"
files = sorted(glob.glob(os.path.join(path, "*")))
sample_files = [f for f in files if "Sample" in os.path.basename(f)]

print(f"Found {len(sample_files)} files\n")
print(f"--- First file: {os.path.basename(sample_files[0])} ---\n")

with open(sample_files[0], 'r') as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i >= 25:
            print("...")
            break
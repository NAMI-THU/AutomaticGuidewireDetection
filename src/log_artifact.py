



#!/usr/bin/env python

import sys



# Get command-line arguments
if len(sys.argv) != 3:
    print("Usage: {} <param1>".format(sys.argv[0]))
    sys.exit(1)

param1 = sys.argv[1]
# param2 = sys.argv[2]

# Your Python code here
print("Parameter 1:", param1)
# print("Parameter 2:", param2)

experiment.log_asset_folder(f'runs/detect/{param1}/')


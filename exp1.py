import os
import argparse
import matplotlib.pyplot as plt
import math
DEBUG = 0

parser = argparse.ArgumentParser()
parser.add_argument('--size', default=False, action='store_true')
parser.add_argument('--footprint', default=False, action='store_true')
args = parser.parse_args()

with open('FB2010-1Hr-150-0.txt', 'r') as ifile:
    lines = ifile.readlines()

# <Number of ports in the fabric> <Number of coflows below (one per line)>
# <Coflow ID> <Arrival time (ms)> <Number of mappers> <Location of map-m> <Number of reducers> <Location of reduce-r:Shuffle megabytes of reduce-r>
sizes = []
is_first_line = True
for line in lines:
    if is_first_line:
        num_coflows = int(line.split()[1])
        is_first_line = False
    else:
        split = line.split()
        num_mappers = int(split[2])
        if DEBUG: print(num_mappers)
        num_reducers = int(split[2+num_mappers+1])
        if DEBUG: print(num_reducers)
        reducers = split[2+num_mappers+2:]
        if DEBUG: print(len(reducers))
        coflow_size = 0
        for reducer in reducers:
            size = float(reducer.split(':')[1])
            coflow_size += size
            if DEBUG: print('coflow size: ', size)
        sizes.append(coflow_size)

sizes = sorted(sizes)

if args.size:
    frac_coflow_x = [sizes[0]]
    frac_coflow_y = [0]
    # coflow size
    for idx in range(len(sizes)):
        size = sizes[idx]
        if size in frac_coflow_x:
            frac_coflow_y[-1] += 1
        else:
            frac_coflow_x.append(size)
            frac_coflow_y.append(frac_coflow_y[-1]+1)
elif args.footprint:
    frac_coflow_x = [sizes[0]]
    frac_coflow_y = [0]

    # coflow footprint
    for size in sizes:
        if size in frac_coflow_x:
            frac_coflow_y[-1] += size
        else:
            frac_coflow_x.append(size)
            frac_coflow_y.append(frac_coflow_y[-1]+size)

# print(frac_coflow_y[-1])
frac_coflow_y = [item / frac_coflow_y[-1] for item in frac_coflow_y] 
# for idx in range(len(frac_coflow_x)):
#     frac_coflow_x[idx] = math.log(frac_coflow_x[idx])
# plt.plot(frac_coflow_x, frac_coflow_y)
plt.semilogx(frac_coflow_x, frac_coflow_y)
plt.show()

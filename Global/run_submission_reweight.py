import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser('Reweight the submission.')
parser.add_argument('--file', type=str)
args = parser.parse_args()

frac = 0.060
a = frac / 0.1835
b = (1 - frac) / ( 1 - 0.1835)
f = lambda x: a* x/(a*x + b * (1 - x))

sub = pd.read_csv(args.file, index_col=0)
sub = sub.rename(columns={sub.columns[0]: 'is_duplicate'})
test_inter = pd.read_csv('test_intersect.csv', index_col=0)
pred = np.zeros_like(sub.is_duplicate, dtype=float)
for i, (flag, x ) in enumerate(zip(test_inter['magic_intersect'] == 0, sub.is_duplicate)):
    if flag:
        pred[i] = f(x)
    else:
        pred[i] = x
    
print(sub.is_duplicate.describe())
sub.is_duplicate = pred
print(sub.is_duplicate.describe())
sub.to_csv(args.file.replace('.csv', '_reweight_%s.csv' % frac))
import numpy as np
import pandas as pd
from itertools import combinations
from itertools import permutations
from itertools import product
from scipy.optimize import linprog
from scipy.stats import norm
import cvxpy as cp
import math
import random
import ast
import matplotlib.pyplot as plt

main = pd.read_csv("/Users/jackadeney/Library/CloudStorage/Dropbox/randomUtility/simulatedData.csv")
check = main.groupby(['f_rational', 'n_alternatives'])[['support', 'rationalWeight', 'irrationalWeight', 'totalWeight']].agg('mean')

check = check.reset_index()

fig, axes = plt.subplots(1, 3, figsize=(12,4))

for ind, n_alt in enumerate(check['n_alternatives'].sort_values().drop_duplicates()):
    dt = check[check['n_alternatives'] == n_alt]
    axes[ind].plot(dt['f_rational'], dt['rationalWeight'])
    axes[ind].plot(dt['f_rational'], dt['irrationalWeight'])
    axes[ind].plot(dt['f_rational'], dt['totalWeight'])
    axes[ind].set_title(f"Number of Alternatives {int(n_alt)}")
    if ind == 0:
        axes[ind].set_ylim(0, 0.2)
    elif ind == 1:
        axes[ind].set_xlabel(f"Proportion of Rational Linear Orders")
        axes[ind].set_ylim(0, 0.06)
    else:
        axes[ind].set_ylim(0, 0.01)


plt.show()
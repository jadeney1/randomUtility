import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

wd_path = "/Users/jackadeney/Library/CloudStorage/Dropbox/randomUtility"

main = pd.read_csv("/Users/jackadeney/Library/CloudStorage/Dropbox/randomUtility/simulatedData.csv")
check = main.groupby(['f_rational', 'n_alternatives'])[['support', 'rationalWeight', 'irrationalWeight', 'totalWeight']].agg('mean')

check = check.reset_index()

fig, axes = plt.subplots(1, 3, figsize=(12,4))

for ind, n_alt in enumerate(check['n_alternatives'].sort_values().drop_duplicates()):
    dt = check[check['n_alternatives'] == n_alt]
    axes[ind].plot(dt['f_rational'], dt['rationalWeight'], label="Rational")
    axes[ind].plot(dt['f_rational'], dt['irrationalWeight'], label="Irrational")
    axes[ind].plot(dt['f_rational'], dt['totalWeight'], label="Total")
    axes[ind].set_title(f"Number of Alternatives {int(n_alt)}")
    if ind == 0:
        axes[ind].set_ylim(0, 0.2)
        axes[ind].set_ylabel(f"Average Weight On Linear Order")
    elif ind == 1:
        axes[ind].set_xlabel(f"Proportion of Rational Linear Orders")
        axes[ind].set_ylim(0, 0.06)
    else:
        axes[ind].set_ylim(0, 0.01)
        axes[ind].legend()

fig.savefig("/Users/jackadeney/Library/CloudStorage/Dropbox/randomUtility/rationalProportions.jpeg")

plt.close()

check2 = main.groupby(['rational', 'f_rational', 'n_alternatives'])[['rationalWeight', 'irrationalWeight', 'totalWeight']].agg('mean').reset_index()
check2['colors'] = check2['rational'].map({True: 'blue', False: 'red'})

fig, axes = plt.subplots(3, 2, figsize=(12,4))
labs = {'rationalWeight': 'Rational Weight', 'irrationalWeight': 'Irrational Weight'}

for j, variable in enumerate(['rationalWeight', 'irrationalWeight']):
    for i, n_alts in enumerate([3,4,5]):
        df_false = check2[~check2['rational'] & (check2['n_alternatives'] == n_alts)]
        df_true = check2[check2['rational'] & (check2['n_alternatives'] == n_alts)]
        axes[i,j].plot(df_false['f_rational'], df_false[variable], color='red', label='Irrational')
        axes[i,j].plot(df_true['f_rational'], df_true[variable], color='blue', label='Rational')
        if j == 0:
            axes[i,j].set_ylabel(f"N Alternatives: {n_alts}")
        if i == 0:
            axes[i,j].set_title(f"{labs[variable]}")
        if (i,j) == (0,1):
            axes[i,j].legend()

fig.savefig("/Users/jackadeney/Library/CloudStorage/Dropbox/randomUtility/weightsByRationality.jpeg")

main['intercept'] = 1
model = sm.Logit(main['rational'], main[['intercept', 'rationalWeight', 'irrationalWeight', 'n_alternatives']])
results = model.fit()
print(results.summary())
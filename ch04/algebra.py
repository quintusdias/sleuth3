import pathlib

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

pd.options.display.float_format = "{:.1f}".format

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0402.csv'
df = pd.read_csv(path)

df['Order'] = df['Time'].rank()
summary = df.groupby('Treatment').describe()
print(summary.loc[:, 'Time'])
print(summary.loc[:, 'Order'])

rbar = df.describe().loc['mean', 'Order']
sr = df.describe().loc['std', 'Order']
n1 = summary.loc['Conventional', ('Time', 'count')]
n2 = summary.loc['Modified', ('Time', 'count')]
mean_t = n1 * rbar
sd_t = sr * np.sqrt(n1 * n2 / (n1 + n2))
T = df.groupby('Treatment').sum().loc['Modified', 'Order']

z = ((T + 0.5) - n1 * rbar) / sd_t
p = scipy.stats.norm.cdf(z)
print(p)

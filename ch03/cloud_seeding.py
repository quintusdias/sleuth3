import pathlib

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

pd.options.display.float_format = "{:.1f}".format

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0301.csv'
df = pd.read_csv(path)

summary = df.groupby('Treatment').describe()
print(summary)
sns.boxplot(x='Treatment', y='Rainfall', data=df)

# Log transform the data.
df['Log Rainfall'] = np.log(df['Rainfall'])
summary = df.groupby('Treatment')['Log Rainfall'].describe()
print(summary)
sns.boxplot(x='Treatment', y='Log Rainfall', data=df)

a = df.loc[df.Treatment == 'Seeded', 'Log Rainfall']
b = df.loc[df.Treatment == 'Unseeded', 'Log Rainfall']
t, p = scipy.stats.ttest_ind(a, b, equal_var=False)

n1 = summary.loc['Seeded', 'count']
n2 = summary.loc['Unseeded', 'count']
s1 = summary.loc['Seeded', 'std']
s2 = summary.loc['Unseeded', 'std']
est = summary.loc['Seeded', 'mean'] - summary.loc['Unseeded', 'mean']
sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
se = sp * np.sqrt(1 / n1 + 1 / n2)
t95 = scipy.stats.t.ppf(0.95, n1 + n2 - 2)

ci = scipy.stats.t.interval(0.95, loc=est, scale=se, df=50)
print(ci)
ci_orig = np.exp(ci)
print(ci_orig)


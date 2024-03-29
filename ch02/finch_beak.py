import pathlib

import numpy as np
import pandas as pd
import scipy.stats

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0201.csv'
orig = pd.read_csv(path)

df = orig.groupby('Year').agg(['count', 'mean', 'std'])
df.index = ['1: Pre-drought', '2: Post-drought']

# Make the column index look nice.
df = df.droplevel(0, axis=1)
columns = {
    'count': 'n',
    'mean': 'Average (mm)',
    'std': 'Sample SD (mm)'
}
df.rename(index=str, columns=columns, inplace=True)
pd.set_option('display.float_format', "{:.4f}".format)
print(df)

est = (df.loc["2: Post-drought", "Average (mm)"] 
       - df.loc["1: Pre-drought", "Average (mm)"])

print('\n')
print(f'Estimate of µ2 - µ1 is {est:.4f}')

# Compute the pooled SD
def pooled_sd(df):
    n1 = df.iloc[0]['n']
    n2 = df.iloc[1]['n']

    sd1 = df.iloc[0]['Sample SD (mm)']
    sd2 = df.iloc[1]['Sample SD (mm)']

    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

sp = pooled_sd(df)
print('\n')
print(f"sp = {sp:.4f}")

# Compute the standard error
def standard_error(df):
    sp = pooled_sd(df)

    n1 = df.iloc[0]['n']
    n2 = df.iloc[1]['n']

    return sp * np.sqrt(1 / n1 + 1 / n2)

se = standard_error(df)
print('\n')
print(f"SE = {se:.4f}")

dof = df.iloc[1]['n'] + df.iloc[0]['n'] - 2
print(f"Degrees of freedom = {dof}")


t = scipy.stats.t.ppf(0.975, dof)
print(f"t(0.975,{dof}) = {t}")
ci = [est - se * t, est + se * t]
print(f"95% Confidence interval = [{ci[0]:.4f}, {ci[1]:.4f}]")

a = orig[orig['Year'] == 1976]['Depth']
b = orig[orig['Year'] == 1978]['Depth']
tstat, p = scipy.stats.ttest_ind(a, b, equal_var=False)     
print(f"t statistic is {tstat:.4f}")
print(f"The two-sided p-value {p:.4f}")

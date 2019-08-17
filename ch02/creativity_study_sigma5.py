import pathlib

import numpy as np
import pandas as pd
import scipy.stats

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0101.csv'
odf = pd.read_csv(path)

# Treatment effect of 5.
odf.loc[odf['Treatment'] == 'Intrinsic', 'Score'] -= 5

df = odf.groupby('Treatment').agg(['count', 'mean', 'std'])

# Make the column index look nice.
df = df.droplevel(0, axis=1)
print(df)

est = (df.loc["Intrinsic", "mean"] - df.loc["Extrinsic", "mean"])

print('\n')
print(f'Estimate of µ2 - µ1 is {est:.4f}')

# Compute the pooled SD
def pooled_sd(df):
    n1 = df.iloc[0]['count']
    n2 = df.iloc[1]['count']

    sd1 = df.iloc[0]['std']
    sd2 = df.iloc[1]['std']

    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

sp = pooled_sd(df)
print('\n')
print(f"sp = {sp:.4f}")

# Compute the standard error
def standard_error(df):
    sp = pooled_sd(df)

    n1 = df.iloc[0]['count']
    n2 = df.iloc[1]['count']

    return sp * np.sqrt(1 / n1 + 1 / n2)

se = standard_error(df)
print('\n')
print(f"SE = {se:.4f}")

dof = df.iloc[1]['count'] + df.iloc[0]['count'] - 2
print(f"Degrees of freedom = {dof}")


t = scipy.stats.t.ppf(0.975, dof)
print(f"t(0.975,{dof}) = {t}")
ci = [est - se * t, est + se * t]
print(f"95% Confidence interval = [{ci[0]:.4f}, {ci[1]:.4f}]")

a = odf[odf['Treatment'] == 'Intrinsic']['Score']
b = odf[odf['Treatment'] == 'Extrinsic']['Score']
tstat, p = scipy.stats.ttest_ind(a, b, equal_var=False)     
print(f"t statistic is {tstat:.4f}")
print(f"The two-sided p-value {p:.4f}")

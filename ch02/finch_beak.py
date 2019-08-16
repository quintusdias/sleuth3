import pathlib

import numpy as np
import pandas as pd

path = pathlib.Path.home() / 'data' / 'sleuth3' / 'case0201.csv'
df = pd.read_csv(path)

summary_df = df.groupby('Year').agg(['count', 'mean', 'std'])
summary_df.index = ['1: Pre-drought', '2: Post-drought']

# Make the column index look nice.
summary_df = summary_df.droplevel(0, axis=1)
columns = {
    'count': 'n',
    'mean': 'Average (mm)',
    'std': 'Sample SD (mm)'
}
summary_df.rename(index=str, columns=columns, inplace=True)
pd.set_option('display.float_format', "{:.4f}".format)
print(summary_df)

# Compute the pooled SD
def pooled_sd(df):
    n1 = df.iloc[0]['n']
    n2 = df.iloc[1]['n']

    sd1 = df.iloc[0]['Sample SD (mm)']
    sd2 = df.iloc[1]['Sample SD (mm)']

    return np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

sp = pooled_sd(summary_df)
print('\n')
print(f"sp = {sp:.4f}")

# Compute the standard error
def standard_error(df):
    sp = pooled_sd(df)

    n1 = df.iloc[0]['n']
    n2 = df.iloc[1]['n']

    return sp * np.sqrt(1 / n1 + 1 / n2)

se = standard_error(summary_df)
print('\n')
print(f"SE = {se:.4f}")


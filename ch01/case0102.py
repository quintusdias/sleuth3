import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

df = pd.read_csv('/Users/jevans/data/sleuth3/case0102.csv')

bins = np.linspace(3800, 8400, 24)

fig, axes = plt.subplots(nrows=2, ncols=1)
df[df.Sex == 'Male'].hist(bins=bins, ax=axes[0])
axes[0].set_ylabel('Males')
axes[0].set_ylim(0, 16)

df[df.Sex == 'Female'].hist(bins=bins, ax=axes[1])
axes[1].set_ylabel('Females')
axes[1].set_ylim(0, 16)
axes[1].set_title('')
axes[1].set_xlabel('Starting Salary ($U.S.)')

fig, ax = plt.subplots()
sns.boxplot(x='Sex', y='Salary', data=df, ax=ax)
ax.set_ylabel('Starting Salary ($U.S.)')
plt.show()


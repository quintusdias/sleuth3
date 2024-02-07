import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

df = pd.read_csv('case1401.csv')

# exploratory analysis

fig, axes = plt.subplots(nrows=1, ncols=2)

default = sns.color_palette()
palette = {k:v for k, v in zip(df['Chimp'].unique(), sns.color_palette()[:4])}

# sort order is lowest sign average to highest
# data = df.sort_values(by='Order', ascending=False)
idx = df.groupby('Sign')['Minutes'].mean(numeric_only=True).sort_values().reset_index().reset_index().set_index('Sign')
fcn = lambda x: idx.loc[x, 'index']
data = df.sort_values(by='Sign', key=fcn)

g1 = sns.lineplot(x='Sign', y='Minutes', hue='Chimp', style='Chimp', markers=True, dashes=False, data=data, palette=palette, ax=axes[0])
g1.set_ylabel('Acquisition Time (min)')

g2 = sns.lineplot(x='Sign', y=np.log(df['Minutes']), hue='Chimp', style='Chimp', markers=True, dashes=False, data=data, legend=False, ax=axes[1], palette=palette)
g2.set_ylabel('Acquisition Time (min): log scale')
g2.yaxis.set_label_position("right")
g2.yaxis.tick_right()


title = 'Coded scatterplots of acquisition times vs. the order number of each Sign'
fig.suptitle(title)

fig.set_size_inches([10, 6])
fig.tight_layout()


# fitting the additive model
formula = 'Minutes ~ C(Chimp, Treatment(reference="Booee")) + C(Sign, Treatment(reference="listen"))'
model = smf.ols(formula=formula, data=df).fit()

## Display 14.6
data = {
    'observed': df['Minutes'],
    'fitted': model.fittedvalues,
    'resid': model.resid,
    'chimp': df['Chimp'],
    'sign': df['Sign']
}
display = (
    pd.DataFrame(data)
      .pivot_table(index='sign', columns='chimp', values=['observed', 'fitted', 'resid'])
      .sort_values(by='sign', key=fcn)
      .stack(level=0)
)

mu = df['Minutes'].mean()

display['average'] = display.mean(axis='columns')
display.loc[(slice(None), 'resid'), 'average'] = np.nan
display.loc[(slice(None), 'fitted'), 'average'] = np.nan
display['effect'] = display['average'] - mu

display.loc[('average', ''), "Booee":"Thelma"] = df.groupby('Chimp')['Minutes'].mean()
display.loc[('average', ''), 'average'] = df['Minutes'].mean()

display.loc[('effect', ''), 'Booee':'Thelma'] = display.loc[('average', ''), 'Booee':'Thelma'] - mu

display.style.format(na_rep='')

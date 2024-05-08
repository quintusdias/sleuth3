# 3rd party library imports
from IPython.display import Markdown as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

sns.set()

df = pd.read_csv('case1601.csv')
df.head()

# aggregate by treatment, then munge the data into long form
_df = (
    df.groupby('Treatment')
      .mean(numeric_only=True)
      .T
      .stack()
      .reset_index()
      .rename({'level_0': 'Week', 0: 'Percentage'}, axis='columns')
)
_df.head()

ax = sns.lineplot(data=_df, x='Week', y='Percentage', hue='Treatment', style='Treatment', markers=True, dashes=False)
ax.invert_xaxis()


def profile_plots(df, fig, gs, title):
    """
    Profile plots for each monkey in treatment group

    Parameters
    ----------
    df : dataframe
    fig : matplotlib figure
    gs : matplotlib gridspec
    title : str
    """

    monkeys = _df['Monkey'].unique()
    
    for idx, monkey in enumerate(monkeys):
        ax = fig.add_subplot(gs[idx])
        data = _df.query('Monkey == @monkey')
        sns.scatterplot(data=data, x='week', y='score', ax=ax)
    
        # draw the long term average
        avg = data.query('week >= 8')['score'].mean()
        x = [8, 12, 16]
        y = [avg, avg, avg]
        h = ax.plot(x, y)
    
        # draw the short term average
        avg = data.query('week < 8')['score'].mean()
        x = [2, 4]
        y = [avg, avg]
        ax.plot(x, y, color=h[0].get_color())
    
        ax.set_xlabel(monkey)
        ax.xaxis.set_label_coords(0.5, 0.10)
        ax.invert_xaxis()
        ax.set_xticklabels([])
    
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.set_ylim(30,100)
    
    # force the same yticks, scale everywhere
    axes = fig.get_axes()
    yticks = [40, 60, 80, 100]
    yticklabels = [str(x) for x in yticks]

    # set yticks on entire left-hand column
    for idx in range(0, len(axes), gs.ncols):
        axes[idx].set_yticks(yticks)
        axes[idx].set_yticklabels(yticklabels)
    
    # set all the xticks properly (not the labels)
    xticks = [2, 4, 8, 12, 16]
    for ax in axes:
        ax.set_xticks(xticks)

    xticklabels = [str(x) for x in _df['week'].unique()]
    # set the xtick labels if axis is on bottom row OR has no siblings below
    # it
    bottom_row = len(axes) // gs.ncols
    for idx, ax in enumerate(axes):
        current_row = idx // gs.ncols 
        if (current_row == bottom_row) or (idx + gs.ncols >= len(axes)):
            axes[idx].set_xticks(xticks)
            axes[idx].set_xticklabels(xticklabels)
    
    fig.suptitle('Control Monkeys')
    fig.tight_layout()

# Do the control monkeys first.
_df = (
    df.query('Treatment=="Control"')
      .drop(labels='Treatment', axis='columns')
      .set_index('Monkey')
      .stack()
      .reset_index(level=1)
      .rename({'level_1': 'week', 0: 'score'}, axis='columns')
      .reset_index()
)
_df['week'] = _df['week'].apply(lambda x: x.replace('Week', '')).astype(int)

_df.head(n=6)

fig = plt.figure(figsize=[10, 6])
gs = fig.add_gridspec(2,4)

profile_plots(_df, fig, gs, 'Control Monkeys')


_df = (
    df.query('Treatment=="Treated"')
      .drop(labels='Treatment', axis='columns')
      .set_index('Monkey')
      .stack()
      .reset_index(level=1)
      .rename({'level_1': 'week', 0: 'score'}, axis='columns')
      .reset_index()
)
_df['week'] = _df['week'].apply(lambda x: x.replace('Week', '')).astype(int)

fig = plt.figure(figsize=[10, 9])
gs = fig.add_gridspec(3,4)

profile_plots(_df, fig, gs, 'Treated Monkeys')

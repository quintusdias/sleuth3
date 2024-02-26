# 3rd party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set()
pd.options.display.float_format = "{:.3f}".format
pd.options.display.max_columns = 12

df = pd.read_csv('case1201.csv')
print(df.head())

cols = ['Rank', 'Takers', 'Years', 'Income', 'Public', 'Expend', 'SAT']
#g = sns.pairplot(df[cols])

# adjust the Takers column, make the SAT/Takers relationship more linear.
df['logTakers'] = np.log(df['Takers'])
#g = sns.pairplot(df)

formula = 'SAT ~ logTakers + Rank'
model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

# partial residual plots
# different from partial regression plots
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
model = smf.ols(formula='SAT ~ logTakers + Rank + Expend', data=df).fit()
sm.graphics.plot_ccpr(model, 'Expend', ax=ax[0])

model = smf.ols(formula='SAT ~ logTakers + Rank + Expend', data=df.query('Expend < 40')).fit()
sm.graphics.plot_ccpr(model, 'Expend', ax=ax[1])


# partial residual plots
# public schools
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
model = smf.ols(formula='SAT ~ logTakers + Rank + Public', data=df).fit()
sm.graphics.plot_ccpr(model, 'Public', ax=ax[0])
ax[0].set_title('With Louisiana')
ax[0].set_ylabel('Partial Residual')
ax[0].set_xlabel('')

model = smf.ols(formula='SAT ~ logTakers + Rank + Public', data=df.query('Public > 50')).fit()
sm.graphics.plot_ccpr(model, 'Public', ax=ax[1])
ax[1].set_title('Without Louisiana')
ax[1].set_ylabel('')
ax[1].set_xlabel('')

fig.suptitle('CCPR (partial residual plots)')
fig.supxlabel('Test Taker Percentage in Public Schools')
fig.tight_layout()

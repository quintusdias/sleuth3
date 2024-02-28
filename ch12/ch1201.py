# standard library imports
import itertools

# 3rd party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


sns.set()
pd.options.display.float_format = "{:.3f}".format
pd.options.display.max_columns = 12
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

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
ax[0].set_title(rf"With Alaska: $\beta$, SE = ({model.params['Expend']:.2f}, {model.bse['Expend']:.2f})")

model = smf.ols(formula='SAT ~ logTakers + Rank + Expend', data=df.query('Expend < 40')).fit()
sm.graphics.plot_ccpr(model, 'Expend', ax=ax[1])
ax[1].set_title(rf"Without Alaska: $\beta$, SE = ({model.params['Expend']:.2f}, {model.bse['Expend']:.2f})")


# partial residual plots
# public schools
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
model = smf.ols(formula='SAT ~ logTakers + Rank + Public', data=df).fit()
sm.graphics.plot_ccpr(model, 'Public', ax=ax[0])
ax[0].set_title(rf"With Louisiana: $\beta$, SE = ({model.params['Public']:.2f}, {model.bse['Public']:.2f})")
ax[0].set_ylabel('Partial Residual')
ax[0].set_xlabel('')

model = smf.ols(formula='SAT ~ logTakers + Rank + Public', data=df.query('Public > 50')).fit()
sm.graphics.plot_ccpr(model, 'Public', ax=ax[1])
ax[1].set_title(rf"Without Louisiana: $\beta$, SE = ({model.params['Public']:.2f}, {model.bse['Public']:.2f})")
ax[1].set_ylabel('')
ax[1].set_xlabel('')

fig.suptitle('CCPR (partial residual plots)')
fig.supxlabel('Test Taker Percentage in Public Schools')
fig.tight_layout()

# Forward Variable Selection
class ForwardSelect(object):

    def __init__(self, df, endog):

        self.endog = endog
        self.df = df

        self.current_exog = []

        self.candidates = df.columns.to_list()
        self.candidates.remove(endog)

    def run_forward(self):
        """
        Go thru each new model where the candidate variables are added to the
        new model exog vars.
        """

        fresults = []

        for var in self.candidates:
            formula = f"{self.endog} ~ {' + '.join(self.current_exog + [var])}"
            model = smf.ols(formula=formula, data=self.df).fit()
            atable = anova_lm(model)
            fresults.append(atable.loc[var, 'F'])

        s = pd.Series(fresults, index=self.candidates)
        print(s)

    def select(self, var):
        """
        Add a variable to the set list of exog vars, probably because of high
        f-value.
        """

        self.candidates.remove(var)
        self.current_exog.append(var)

    def __str__(self):
        if len(self.current_exog) == 0:
            formula = f"{self.endog} ~ 1"
        else:
            formula = f"{self.endog} ~ {' + '.join(self.current_exog)}"
        model = smf.ols(formula=formula, data=self.df).fit()
        return str(model.summary())


data = df.query('Expend < 40').drop(['State', 'Takers'], axis='columns')
fs = ForwardSelect(data, 'SAT')
fs.run_forward()

fs.select('logTakers')
fs.run_forward()

fs.select('Expend')
fs.run_forward()

print(fs)

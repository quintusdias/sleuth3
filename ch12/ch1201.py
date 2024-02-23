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


# Match the plot in the book
formula = 'SAT ~ logTakers + Rank + Expend'

model1 = smf.ols(formula=formula, data=df).fit()
x1, y1 = df['Expend'], model1.resid + model1.params['Expend'] * df['Expend']

df2 = df.query('State != "Alaska"')
model2 = smf.ols(formula=formula, data=df2).fit()
x2, y2 = df2['Expend'], model2.resid + model2.params['Expend'] * df2['Expend']


fig, ax = plt.subplots()
sns.scatterplot(x=x1, y=y1, ax=ax)
sns.scatterplot(x=x2, y=y2, ax=ax)

ax.set_ylabel('Partial Residual')
ax.set_xlabel('Expenditure ($100s per student)')


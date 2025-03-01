import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_ind

data = pd.read_csv('your_data.csv')

print(data.info())
print(data.describe())

sns.histplot(data['eval'], bins=20, kde=True)
plt.title('Distribution of Teacher Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

tenured_eval = data[data['tenured_prof'] == 1]['eval']
non_tenured_eval = data[data['tenured_prof'] == 0]['eval']
t_stat, p_value = ttest_ind(tenured_eval, non_tenured_eval)
print(f'T-test: t-stat={t_stat:.2f}, p-value={p_value:.4f}')

X = sm.add_constant(data[['tenured_prof', 'vismin']])
y = data['eval']
model = sm.OLS(y, X).fit()
print(model.summary())

if p_value < 0.05:
    print('There is a statistically significant difference in ratings between tenured and non-tenured professors.')
else:
    print('There is no significant difference in teacher ratings based on tenure status.')

print('The regression analysis shows that some factors, such as professor status and belonging to a visible minority, may influence the ratings.')

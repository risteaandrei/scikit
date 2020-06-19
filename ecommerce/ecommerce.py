import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

customers = pd.read_csv('ecommerce/ecommerce_customers.csv')
#print(customers.columns)

x = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(x_train, y_train)

predictions = lm.predict(x_test)
cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
print(cdf)
#print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
#print('MSE: ', metrics.mean_squared_error(y_test, predictions))
#print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#plt.scatter(y_test, predictions)
#plt.xlabel('Y Test (True Values)')
#plt.ylabel('Predicted Values')
#sns.distplot((y_test-predictions), bins=50)
#plt.show()

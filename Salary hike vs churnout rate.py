# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

# Input Variable = Salary Hike
# Output Varibale = Churn out rate

emp_data= pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/emp_data.csv")
emp_data.columns = "sal_hike", "churnout_rate"
emp_data.columns

# Exploratory data analysis:
emp_data.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = emp_data.churnout_rate, x = np.arange(1,11, 1))
plt.hist(emp_data.churnout_rate) # Histogram
plt.boxplot(emp_data.churnout_rate) # Boxplot

plt.bar(height = emp_data.sal_hike, x = np.arange(1,11, 1))
plt.hist(emp_data.sal_hike) # Histogram
plt.boxplot(emp_data.sal_hike) # Boxplot

# Scatter plot
plt.scatter(x = emp_data['sal_hike'], y = emp_data['churnout_rate'], color = 'red') 

# Correlation
np.corrcoef(emp_data.sal_hike, emp_data.churnout_rate) 

# Covariance
cov_output = np.cov(emp_data.sal_hike, emp_data.churnout_rate)[0, 1]
cov_output


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('churnout_rate ~ sal_hike', data = emp_data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp_data['sal_hike']))

# Regression Line
plt.scatter(emp_data.sal_hike, emp_data.churnout_rate)
plt.plot(emp_data.sal_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = emp_data.churnout_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model building on Transformed Data(Log Transformation)
# x = log(Salary Hike); y = Churn out rate

plt.scatter(x = np.log(emp_data['sal_hike']), y = emp_data['churnout_rate'], color = 'brown')
np.corrcoef(np.log(emp_data.sal_hike), emp_data.churnout_rate) 

model2 = smf.ols('churnout_rate ~ np.log(sal_hike)', data = emp_data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp_data['sal_hike']))

# Regression Line
plt.scatter(np.log(emp_data.sal_hike), emp_data.churnout_rate)
plt.plot(np.log(emp_data.sal_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = emp_data.churnout_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation
# x = Salary Hike; y = log(Churn out rate)

plt.scatter(x = emp_data['sal_hike'], y = np.log(emp_data['churnout_rate']), color = 'orange')
np.corrcoef(emp_data.sal_hike, np.log(emp_data.churnout_rate)) 

model3 = smf.ols('np.log(churnout_rate) ~ sal_hike', data = emp_data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp_data['sal_hike']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(emp_data.sal_hike, np.log(emp_data.churnout_rate))
plt.plot(emp_data.sal_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = emp_data.churnout_rate - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation
# x = sal_hike; x^2 = sal_hike*sal_hike; y = log(churnout_rate)

model4 = smf.ols('np.log(churnout_rate) ~ sal_hike + I(sal_hike*sal_hike)', data = emp_data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp_data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = emp_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


plt.scatter(emp_data.sal_hike, np.log(emp_data.churnout_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = emp_data.churnout_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# Analysis - Polynomial model is the best fitted model with Low Rmse

from sklearn.model_selection import train_test_split

train, test = train_test_split(emp_data, test_size = 0.2)

finalmodel = smf.ols('np.log(churnout_rate) ~ sal_hike + I(sal_hike*sal_hike )', data = emp_data).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_churnout_rate = np.exp(test_pred)
pred_test_churnout_rate

# Model Evaluation on Test data
test_res = test.churnout_rate - pred_test_churnout_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_churnout_rate = np.exp(train_pred)
pred_train_churnout_rate

# Model Evaluation on train data
train_res = train.churnout_rate - pred_train_churnout_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

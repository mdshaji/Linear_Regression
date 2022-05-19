# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

# Input Variable = Years of Experience
# Output Varibale = Salary Hike

Sal_hike = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/Salary_Data.csv")
Sal_hike.columns = "Experience", "Salary" 
Sal_hike.columns

# Exploratory data analysis:
Sal_hike.describe()

# Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = Sal_hike.Salary, x = np.arange(1, 31, 1))
plt.hist(Sal_hike.Salary) #histogram
plt.boxplot(Sal_hike.Salary) #boxplot

plt.bar(height = Sal_hike.Experience, x = np.arange(1, 31, 1))
plt.hist(Sal_hike.Experience) #histogram
plt.boxplot(Sal_hike.Experience) #boxplot

# Scatter plot
plt.scatter(x = Sal_hike['Experience'], y = Sal_hike['Salary'], color = 'red') 

# Correlation
np.corrcoef(Sal_hike.Experience, Sal_hike.Salary) 

# Covariance
cov_output = np.cov(Sal_hike.Experience, Sal_hike.Salary)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Salary ~ Experience', data = Sal_hike).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(Sal_hike['Experience']))

# Regression Line
plt.scatter(Sal_hike.Experience, Sal_hike.Salary)
plt.plot(Sal_hike.Experience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = Sal_hike.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model building on Transformed Data(Log Transformation)
# x = log(Experience); y = Salary

plt.scatter(x = np.log(Sal_hike['Experience']), y = Sal_hike['Salary'], color = 'brown')
np.corrcoef(np.log(Sal_hike.Experience), Sal_hike.Salary) 

model2 = smf.ols('Salary ~ np.log(Experience)', data = Sal_hike).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(Sal_hike['Experience']))

# Regression Line
plt.scatter(np.log(Sal_hike.Experience), Sal_hike.Salary)
plt.plot(np.log(Sal_hike.Experience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = Sal_hike.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation
# x = Experience; y = log(Salary)

plt.scatter(x = Sal_hike['Experience'], y = np.log(Sal_hike['Salary']), color = 'orange')
np.corrcoef(Sal_hike.Experience, np.log(Sal_hike.Salary))

model3 = smf.ols('np.log(Salary) ~ Experience', data = Sal_hike).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(Sal_hike['Experience']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(Sal_hike.Experience, np.log(Sal_hike.Salary))
plt.plot(Sal_hike.Experience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = Sal_hike.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation
# x = Experience; x^2 = Experience*Experience; y = log(Salary)

model4 = smf.ols('np.log(Salary) ~ Experience + I(Experience*Experience)', data = Sal_hike).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(Sal_hike))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = Sal_hike.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


plt.scatter(Sal_hike.Experience, np.log(Sal_hike.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = Sal_hike.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


# Analysis - SLR is the best fitted model with Low Rmse

from sklearn.model_selection import train_test_split

train, test = train_test_split(Sal_hike, test_size = 0.2)

finalmodel = smf.ols('Salary ~ Experience', data = train).fit()
finalmodel.summary()


# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
test_res = test.Salary - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
train_pred

# Model Evaluation on train data

train_res = train.Salary -  train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse



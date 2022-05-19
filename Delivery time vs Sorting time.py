# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

# Input Variable  [x] = Sorting Time
# Output Variable [y]= Delivery Time

del_time = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/delivery_time.csv")
del_time.columns = "delivery_time", "Sort_time"
del_time.columns

# Exploratory data analysis:
del_time.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = del_time.delivery_time, x = np.arange(1,22, 1))
plt.hist(del_time.delivery_time) # Histogram 
plt.boxplot(del_time.delivery_time) # Boxplot

plt.bar(height = del_time.Sort_time, x = np.arange(1, 22, 1))
plt.hist(del_time.Sort_time) # Histogram
plt.boxplot(del_time.Sort_time) # Boxplot

# Scatter plot
plt.scatter(x = del_time['Sort_time'], y = del_time['delivery_time'], color = 'red') 

# Correlation
np.corrcoef(del_time.Sort_time, del_time.delivery_time) 

# Covariance
cov_output = np.cov(del_time.Sort_time, del_time.delivery_time)[0, 1]
cov_output


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('delivery_time ~ Sort_time', data = del_time).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(del_time['Sort_time']))

# Regression Line
plt.scatter(del_time.Sort_time, del_time.delivery_time)
plt.plot(del_time.Sort_time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = del_time.delivery_time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model building on Transformed Data(Log Transformation)
# x = log(Sorting Time); y = Delivery Time

plt.scatter(x = np.log(del_time['Sort_time']), y = del_time['delivery_time'], color = 'brown')
np.corrcoef(np.log(del_time.Sort_time), del_time.delivery_time)

model2 = smf.ols('delivery_time ~ np.log(Sort_time)', data = del_time).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(del_time['Sort_time']))

# Regression Line
plt.scatter(np.log(del_time.Sort_time), del_time.delivery_time)
plt.plot(np.log(del_time.ST), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = del_time.delivery_time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation
# x = Sorting Time; y = log(Delivery Time)

plt.scatter(x = del_time['Sort_time'], y = np.log(del_time['delivery_time']), color = 'orange')
np.corrcoef(del_time.Sort_time, np.log(del_time.delivery_time))

model3 = smf.ols('np.log(delivery_time) ~ Sort_time', data = del_time).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(del_time['Sort_time']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(del_time.Sort_time, np.log(del_time.delivery_time))
plt.plot(del_time.Sort_time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = del_time.delivery_time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation
# x = Sort_time; x^2 = Sort_time*Sort_time; y = log(delivery_time)

model4 = smf.ols('np.log(delivery_time) ~ Sort_time + I(Sort_time*Sort_time)', data = del_time).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(del_time))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = del_time.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)


plt.scatter(del_time.Sort_time, np.log(del_time.delivery_time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = del_time.delivery_time - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


# Analysis - Log Model 1 is the best fitted model with Low Rmse

from sklearn.model_selection import train_test_split

train, test = train_test_split(del_time, test_size = 0.2)

finalmodel = smf.ols('delivery_time ~ np.log(Sort_time)', data = train).fit()
finalmodel.summary()


# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred

# Model Evaluation on Test data
test_res = test.delivery_time - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
train_pred

# Model Evaluation on train data
train_res = train.delivery_time -  train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse



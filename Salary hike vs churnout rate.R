# Input Variable = Salary Hike
# Output Varibale = Churn out rate

#Importing Dataset

Emp.data <- read.csv(file.choose(), header = T)
colnames(Emp.data) <- c("sal_hike", "churnout_rate") 
View(Emp.data)
attach(Emp.data)

# Exploratory data analysis
summary(Emp.data)


install.packages("Hmisc")
library(Hmisc)
describe(Emp.data) # To have a quick glance of a data


install.packages("lattice") # Data Visualisation
library("lattice") 

# Graphical exploration
dotplot(Emp.data$sal_hike, main = "Dot Plot of Salary Hike")
dotplot(Emp.data$churnout_rate, main = "Dot Plot of Churn Out Rate")

#Boxplot Representation

boxplot(Emp.data$sal_hike, col = "dodgerblue4",main = "Salary Hike")
boxplot(Emp.data$churnout_rate, col = "red", horizontal = T, main = "Churn Out Rate")

#Histogram Representation

hist(Emp.data$sal_hike,main = "Salary Hike")
hist(Emp.data$churnout_rate, main = "Churn Out Rate")

# Normal QQ plot
qqnorm(Emp.data$sal_hike,main = "Salary Hike")
qqline(Emp.data$sal_hike,main = "Salary Hike")

qqnorm(Emp.data$churnout_rate , main = "Churn Out Rate")
qqline(Emp.data$churnout_rate , main = "Churn Out Rate")


# Bivariate analysis
# Scatter plot
plot(Emp.data$sal_hike, Emp.data$churnout_rate, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Salary Hike", 
     ylab = "Churn Out Rate", pch = 20)  # plot(x,y)


# Correlation Coefficient
cor(sal_hike,churnout_rate)

# Linear Regression model
reg <- lm(churnout_rate ~ sal_hike, data = Emp.data ) # Y ~ X
summary(reg)

confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

View(pred)

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = Emp.data, aes(sal_hike, churnout_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)


# Evaluation the model for fitness 
cor(pred$fit, Emp.data$churnout_rate)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques(transform the variables to check whether the predicted values are better)
# input = log(x); output = y

plot(log(sal_hike), churnout_rate)
cor(log(sal_hike), churnout_rate)

reg_log <- lm(churnout_rate ~ log(sal_hike), data = Emp.data)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, Emp.data$churnout_rate)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = Emp.data, aes(log(sal_hike), churnout_rate) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))



# Log transformation(y)
# input = x; output = log(y)

plot(sal_hike, log(churnout_rate))
cor(sal_hike, log(churnout_rate))

reg_log1 <- lm(log(churnout_rate) ~ sal_hike ,data= Emp.data)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, Emp.data$churnout_rate)

res_log1 = churnout_rate - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = Emp.data, aes(sal_hike, log(churnout_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)



# Polynomial models(Non Linear)
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(churnout_rate) ~ sal_hike + I(sal_hike*sal_hike), data = Emp.data)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, Emp.data$churnout_rate)

res2 = churnout_rate - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = Emp.data, aes(sal_hike, log(churnout_rate)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))


# Data Partition
# Random Sampling
n <- nrow(Emp.data)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- Emp.data[train_ind, ]
test <-  Emp.data[-train_ind, ]

plot(train$sal_hike, log(train$churnout_rate))
plot(test$sal_hike, log(test$churnout_rate))

model <- lm(log(churnout_rate) ~ sal_hike + I(sal_hike*sal_hike), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$churnout_rate - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model,interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$churnout_rate - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse

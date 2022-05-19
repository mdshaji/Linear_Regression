# Input Variable = Years of Experience
# Output Varibale = Salary Hike

#Importing Dataset

Sal_hike <- read.csv(file.choose(), header = T)
colnames(Sal_hike) <- c("Experience", "Salary") 
View(Sal_hike)
attach(Sal_hike)

# Exploratory data analysis
summary(Sal_hike)

install.packages("Hmisc")
library(Hmisc)
describe(Sal_hike) # To have a quick glance of a data

install.packages("lattice")
library("lattice")

# Graphical exploration
dotplot(Sal_hike$Experience, main = "Dot Plot of Experience")
dotplot(Sal_hike$Salary, main = "Dot Plot of Salary")

#Boxplot Representation

boxplot(Sal_hike$Experience, col = "dodgerblue4",main = "Experience")
boxplot(Sal_hike$Salary, col = "red", horizontal = T, main = "Salary")

#Histogram Representation

hist(Sal_hike$Experience,main = "Experience")
hist(Sal_hike$Salary, main = "Salary")

# Normal QQ plot
qqnorm(Sal_hike$Experience,main = "Experience")
qqline(Sal_hike$Experience,main = "Experience")

qqnorm(Sal_hike$Salary, main = "Salary")
qqline(Sal_hike$Salary, main = "Salary")


# Bivariate analysis
# Scatter plot
plot(Sal_hike$Experience, Sal_hike$Salary, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Experience", 
     ylab = "Salary", pch = 20)  # plot(x,y)

# Correlation Coefficient
cor(Experience,Salary)

# Linear Regression model
reg <- lm(Salary ~ Experience, data = Sal_hike ) # Y ~ X
summary(reg)

confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

View(pred)

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = Sal_hike, aes(Experience, Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)

# Evaluation the model for fitness 
cor(pred$fit, Sal_hike$Salary)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse

# Transformation Techniques(transform the variables to check whether the predicted values are better)
# input = log(x); output = y

plot(log(Experience), Salary)
cor(log(Experience), Salary)

reg_log <- lm(Salary ~ log(Experience), data = Sal_hike)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, Sal_hike$Salary)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = Sal_hike, aes(log(Experience), Salary) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Log transformation(y)
# input = x; output = log(y)

plot(Experience, log(Salary))
cor(Experience, log(Salary))

reg_log1 <- lm(log(Salary) ~ Experience ,data= Sal_hike)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Exponential function for antilog
pred <- as.data.frame(pred)
cor(pred$fit, Sal_hike$Salary)

res_log1 = Salary - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = Sal_hike, aes(Experience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)

# Polynomial models(Non Linear)
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(Salary) ~ Experience + I(Experience*Experience), data = Sal_hike)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, Sal_hike$Salary)

res2 = Salary - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = Sal_hike, aes(Experience, log(Salary)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Data Partition(Randon Sampling)

n <- nrow(Sal_hike)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- Sal_hike[train_ind, ]
test <-  Sal_hike[-train_ind, ]

plot(train$Experience, log(train$Salary))
plot(test$Experience, log(test$Salary))

model <- lm(log(Salary) ~ Experience + I(Experience*Experience), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)

predict_original <- exp(log_res) # converting log values to original values
predict_original <- as.data.frame(predict_original)
test_error <- test$Salary - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model,interval = "confidence", newdata = train)

predict_original_train <- exp(log_res_train) # converting log values to original values
predict_original_train <- as.data.frame(predict_original_train)
train_error <- train$Salary - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse

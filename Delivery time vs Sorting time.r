# INPUT(X) = Sorting Time
# OUTPUT(Y) = Delivery Time


deltime <- read.csv(file.choose(), header = T)
colnames(deltime) <- c("dt", "sorttime") # dt = delivery time & sorttime = sorting time
View(deltime)

# Exploratory data analysis
summary(deltime)

install.packages("Hmisc")
library(Hmisc)
describe(deltime)
?describe

install.packages("lattice")
library("lattice") 

# Graphical exploration
dotplot(deltime$sorttime, main = "Dot Plot of Sorting Time")
dotplot(deltime$dt, main = "Dot Plot of Delivery Time")

?boxplot
boxplot(deltime$sorttime, col = "dodgerblue4",main = "Sorting Time")
boxplot(deltime$dt, col = "red", horizontal = T, main = "Delivery Time")

hist(deltime$sorttime,main = "Sorting Time")
hist(deltime$dt,main = "Delivery Time")

# Normal QQ plot
qqnorm(deltime$sorttime,main = "Sorting Time")
qqline(deltime$sorttime,main = "Sorting Time")

qqnorm(deltime$dt,main = "Delivery Time")
qqline(deltime$dt,main = "Delivery Time")

hist(deltime$st, prob = TRUE)            # prob=TRUE for probabilities not counts
lines(density(deltime$sorttime))             # add a density estimate with defaults
lines(density(deltime$sorttime, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(deltime$dt, prob = TRUE)            # prob=TRUE for probabilities not counts
lines(density(deltime$dt))             # add a density estimate with defaults
lines(density(deltime$dt, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(deltime$sorttime, deltime$dt, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Sorting Time", 
     ylab = "Delivery Time", pch = 20)  # plot(x,y)

attach(deltime)

# Correlation Coefficient
cor(sorttime,dt)

# Linear Regression model
reg <- lm(dt ~ sorttime, data = deltime) # Y ~ X
summary(reg)

confint(reg, level = 0.95)


pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

View(pred)
?predict

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = deltime, aes(sorttime, dt) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)


# Evaluation the model for fitness 
cor(pred$fit, deltime$dt)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques(transform the variables to check whether the predicted values are better)
#input = log(x); output = y

plot(log(sorttime), dt)
cor(log(sorttime), dt)

reg_log <- lm(dt ~ log(sorttime), data = deltime)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, deltime$dt)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = deltime, aes(log(sorttime), dt) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))


# Log transformation(y)
# input = x; output = log(y)

plot(sorttime, log(dt))
cor(sorttime, log(dt))

reg_log1 <- lm(log(dt) ~ sorttime, data = deltime)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, deltime$dt)

res_log1 = dt - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = deltime, aes(sorttime, log(dt)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)


# Polynomial models(Non Linear)
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(dt) ~ sorttime + I(sorttime*sorttime), data = deltime)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, deltime$dt)

res2 = dt - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = deltime, aes(sorttime, log(dt)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Data Partition(Random Sampling)
n <- nrow(deltime)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- deltime[train_ind, ]
test <-  deltime[-train_ind, ]

plot(train$sorttime, train$dt)
plot(test$sorttime, test$dt)

model <- lm(dt ~ log(sorttime), data = train)
summary(model)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- as.data.frame(log_res)
test_error <- test$dt - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model,interval = "confidence", newdata = train)

predict_original_train <- as.data.frame(log_res_train)
train_error <- train$dt - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse

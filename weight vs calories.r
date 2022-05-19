# INPUT(X) = CALORIES CONSUMED
# OUTPUT(Y) = WEIGHGT GAINED


wg.cc <- read.csv(file.choose(), header = T)
colnames(wg.cc) <- c("Weight gain", "Calories")
View(wg.cc)

# Exploratory data analysis
summary(wg.cc)

install.packages("Hmisc")
library(Hmisc)
describe(wg.cc)
?describe

install.packages("lattice")
library("lattice") # dotplot is part of lattice package

# Graphical exploration
dotplot(wg.cc$Calories, main = "Dot Plot of Calories Consumed")
dotplot(wg.cc$`Weight gain`, main = "Dot Plot of Weight Gained")

?boxplot
boxplot(wg.cc$Calories, col = "dodgerblue4")
boxplot(wg.cc$`Weight gain`, col = "red", horizontal = T)

hist(wg.cc$Calories)
hist(wg.cc$`Weight gain`)

# Normal QQ plot
qqnorm(wg.cc$Calories)
qqline(wg.cc$Calories)

qqnorm(wg.cc$`Weight gain`)
qqline(wg.cc$`Weight gain`)

hist(wg.cc$Calories, prob = TRUE)            # prob=TRUE for probabilities not counts
lines(density(wg.cc$Calories))             # add a density estimate with defaults
lines(density(wg.cc$Calories, adjust = 2), lty = "dotted")   # add another "smoother" density

hist(wg.cc$`Weight gain`, prob = TRUE)            # prob=TRUE for probabilities not counts
lines(density(wg.cc$`Weight gain`))             # add a density estimate with defaults
lines(density(wg.cc$`Weight gain`, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(wg.cc$Calories, wg.cc$`Weight gain`, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Calories Consumed", 
     ylab = "Weight Gained", pch = 20)  # plot(x,y)

?plot

## alternate simple command
plot(wg.cc$Calories, wg.cc$`Weight gain`)

attach(wg.cc)

# Correlation Coefficient
cor(Calories, `Weight gain`)

# Linear Regression model
reg <- lm(`Weight gain` ~ Calories, data = wg.cc) # Y ~ X
?lm
summary(reg)

confint(reg, level = 0.95)
?confint

pred <- predict(reg, interval = "predict")
pred <- as.data.frame(pred)

View(pred)
?predict

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = wg.cc, aes(Calories, `Weight gain`) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ x)


# Evaluation the model for fitness 
cor(pred$fit, wg.cc$`Weight gain`)

reg$residuals
rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques
# transform the variables to check whether the predicted values are better

#input = log(x); output = y

plot(log(Calories), `Weight gain`)
cor(log(Calories), `Weight gain`)

reg_log <- lm(`Weight gain` ~ log(Calories), data = wg.cc)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit, wg.cc$`Weight gain`)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse

# Regression line for data
ggplot(data = wg.cc, aes(log(Calories), `Weight gain`) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ log(x))

# Alternate way
ggplot(data = wg.cc, aes(x = log(Calories), y = `Weight gain`)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = wg.cc, aes(x = log(Calories), y = pred$fit))



# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Calories, log(`Weight gain`))
cor(Calories, log(`Weight gain`))

reg_log1 <- lm(log(`Weight gain`) ~ Calories, data = wg.cc)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)
cor(pred$fit, wg.cc$`Weight gain`)

res_log1 = `Weight gain` - pred$fit
rmse <- sqrt(mean(res_log1^2))
rmse

# Regression line for data
ggplot(data = wg.cc, aes(Calories, log(`Weight gain`)) ) +
  geom_point() + stat_smooth(method = lm, formula = log(y) ~ x)


# Polynomial models(Non Linear)
# input = x & x^2 (2-degree) and output = log(y)

reg2 <- lm(log(`Weight gain`) ~ Calories + I(Calories*Calories), data = wg.cc)
summary(reg2)

predlog <- predict(reg2, interval = "predict")
pred <- exp(predlog)

pred <- as.data.frame(pred)
cor(pred$fit, wg.cc$`Weight gain`)

res2 = `Weight gain` - pred$fit
rmse <- sqrt(mean(res2^2))
rmse

# Regression line for data
ggplot(data = wg.cc, aes(Calories, log(`Weight gain`)) ) +
  geom_point() + stat_smooth(method = lm, formula = y ~ poly(x, 2, raw = TRUE))

# Data Partition(Random Sampling)
n <- nrow(wg.cc)
n1 <- n * 0.8
n2 <- n - n1

train_ind <- sample(1:n, n1)
train <- wg.cc[train_ind, ]
test <-  wg.cc[-train_ind, ]

plot(train$Calories, train$`Weight gain`)
plot(test$Calories, test$`Weight gain`)

model <- lm(`Weight gain` ~ Calories, data = wg.cc) # Y ~ X
summary(reg)

confint(model,level=0.95)

log_res <- predict(model,interval = "confidence", newdata = test)
predict_original <- as.data.frame(log_res)
test_error <- test$`Weight gain` - predict_original$fit # calculate error/residual
test_error

test_rmse <- sqrt(mean(test_error^2))
test_rmse

log_res_train <- predict(model,interval = "confidence", newdata = train)

predict_original_train <- as.data.frame(log_res_train)
train_error <- train$`Weight gain` - predict_original_train$fit # calculate error/residual
train_error

train_rmse <- sqrt(mean(train_error^2))
train_rmse


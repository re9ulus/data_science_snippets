# Logistic regression
# Solution to Kaggle Titanic problem, with result 0.74641
# Using the materials of course: MITx: 15.071x The Analytics Edge

setwd("Path/To/Data")
data = read.csv("train.csv")
test = read.csv("test.csv")
str(data)

# Clear data
data$Age[is.na(data$Age)] <- median(data$Age, na.rm=TRUE)
test$Age[is.na(test$Age)] <- median(test$Age, na.rm=TRUE)

# Create model
model = glm(Survived ~ Sex + Pclass + Age, data=data, family = binomial)
summary(model)

# Get prediction
predictTrain = predict(model, newdata = test, type="response")
summary(predictTrain)
test$Survived[predictTrain >= 0.5] = 1
test$Survived[predictTrain < 0.5] = 0

submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "result.csv", row.names = FALSE)

predictTrain = predict(model, type="response")
tapply(predictTrain, data$Survived, mean)

# Confusion matrix with threshold value = 0.5
table(data$Survived, predictTrain > 0.5)

## ROC Curve
library(ROCR)
ROCRpred = prediction(predictTrain, data$Survived)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf)
plot(ROCRperf, colorize=TRUE)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0, 1, 0.1), text.adj=c(-0.2, 1.7))

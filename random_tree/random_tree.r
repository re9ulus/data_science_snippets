# Random Tree (CART), Cross-validation, Random forest
# Solution to Kaggle Titanic problem, with result 0.79426
# Using the materials of course: MITx: 15.071x The Analytics Edge

# Installing package 
#install.packages("caTools")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("rattle")
#install.packages("RColorBrewer")
library(caTools)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)

# Prepare data
setwd("F:/Dev/data_science")
data = read.csv("train.csv")
test = read.csv("test.csv")

str(data)
set.seed(3000)

data$Age[is.na(data$Age)] <- median(data$Age, na.rm=TRUE)
data$Pclass <- as.factor(data$Pclass)
data$Survived <- as.factor(data$Survived)

test$Age[is.na(test$Age)] <- median(test$Age, na.rm=TRUE)
test$Pclass <- as.factor(test$Pclass)

### Building decision tree model (CART), result 0.79426

TitanicTree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
                     data = data, method = "class",
                     control=rpart.control(minbucket = 25))

# Plot tree
# base plot
prp(TitanicTree)

# pretty plot
fancyRpartPlot(TitanicTree) 

# Predicitons
PredictCART <- predict(TitanicTree, newdata = test, type="class")
test$Survived <- PredictCART
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "result.csv", row.names = FALSE)


### Cross-validation, result 0.76555

#install.packages("caret")
#install.packages("e1071")
library(caret)
library(e1071)

# !IMPORTANT: Outcome variable should be a factor

fitControl <- trainControl(method="cv", number=30)
cartGrid <- expand.grid(.cp=(1:200)*0.001)

train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data=data, method="rpart",
      trControl=fitControl, tuneGrid=cartGrid)

TitanicTree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
                     data = data, method = "class",
                     control=rpart.control(cp = 0.004))

fancyRpartPlot(TitanicTree)

# Predicitons
PredictCART <- predict(TitanicTree, newdata = test, type="class")
test$Survived <- PredictCART
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "result.csv", row.names = FALSE)


### Random forest, result 0.77033

# install package
#install.packages("randomForest")
library(randomForest)

# Building model
# !IMPORTANT: Outcome variable should be a factor

# nodesize - number of items in bucket; ntree - number of trees to build
TrainForest <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
                            data=data, nodesize=25, ntree=200)

# Predicitons
PredictForest <- predict(TrainForest, newdata = test)
test$Survived <- PredictForest
test$Survived[is.na(test$Survived)] <- 0 # One value become NA
test$Survived
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "result.csv", row.names = FALSE)


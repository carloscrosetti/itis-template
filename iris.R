# Code from https://machinelearningmastery.com/machine-learning-in-r-step-by-step/
# Author: Jason Brownlee
#
# WARNING set the directory below to the place where you downloaded
# iris.csv and iris.R
#
# Minor fixes added by Carlos Crosetti (carlos.crosetti@outlook.com)
#
# 12/25/2017 - added clc() function to clear the console
# 06/17/2018 - added packages randomForest, lattice and ellipse
# 
# Tested with R 3.5.0
#

clc <- function() cat(rep("\n",50))
clc()
print(paste("R version ", R.version$major, R.version$minor))

setwd("C:/Users/Carlos/OneDrive/DS/2017 Iris Template")
getwd()

Sys.sleep(3)

# WARNING before running this script make sure you installed these
# three packages from the console

# install.packages("caret")
# install.packages("e1071")
# install.packages("tidyr")
# install.packages("randomForest")
# install.packages("lattice")
# install.packages("ellipse")

library(tidyr)
library(caret)
library(e1071)

# this allows to scroll the graphics device
windows(record=TRUE)

Sys.sleep(3)

# define the filename
filename <- "iris.csv"

# load the CSV file from the local directory
dataset <- read.csv(filename, header=FALSE)

# TODO limit the data set to some X rows to develop the model with less data

# set the column names in the dataset
colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# dimensions of dataset
dim(dataset)

# show dataset head
head(dataset)

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)

Sys.sleep(3)

# list the levels for the class
levels(dataset$Species)

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

start_time <- Sys.time()

# summarize attribute distributions
summary(dataset)

end_time <- Sys.time()
end_time - start_time

Sys.sleep(3)

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

Sys.sleep(3)

# barplot for class breakdown
plot(y)

Sys.sleep(3)

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

Sys.sleep(3)

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

Sys.sleep(3)

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

Sys.sleep(3)

start_time <- Sys.time()

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

print("Time spent in training")
end_time <- Sys.time()
end_time - start_time

Sys.sleep(3)

# compare accuracy of models
dotplot(results)

Sys.sleep(3)

# summarize Best Model
print(fit.lda)

Sys.sleep(3)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

# end of iris.R script

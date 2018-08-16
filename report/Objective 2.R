library(dplyr)
library(data.table)

#Read the train and test data
train <- fread("https://raw.githubusercontent.com/VolodymyrOrlov/MSDS6372_Project2/master/data/bank-balanced-train.csv", stringsAsFactors = TRUE)
test <- fread("https://raw.githubusercontent.com/VolodymyrOrlov/MSDS6372_Project2/master/data/bank-test.csv", stringsAsFactors = TRUE)

#Remove first column:

trainLR <- train[,2:18]
testLR <- test[,2:18]

#Age: divide into 3 age groups
trainLR$adult <- ifelse(trainLR$age <= 35, 1, 0)
trainLR$middleaged <- ifelse(trainLR$age >= 36 & trainLR$age <= 60, 1, 0)
trainLR$elderly <- ifelse(trainLR$age > 60, 1, 0)
testLR$adult <- ifelse(testLR$age <= 35, 1, 0)
testLR$middleaged <- ifelse(testLR$age >= 36 & testLR$age <= 60, 1, 0)
testLR$elderly <- ifelse(testLR$age > 60, 1, 0)

#Balance: we will create new variables to indicate negative, positive and zero balance. 
trainLR$balance_pos <- ifelse(train$balance > 0 & train$balance <= 100, "0 to 100",
                       ifelse(train$balance > 100 & train$balance <= 500, "100 to 500",
                       ifelse(train$balance >500 & train$balance <= 2000, "500 to 2000",
                       ifelse(train$balance >2000 & train$balance <= 10000, "2000 to 10000",
                       ifelse(train$balance >10000, "more than 10000",
                       ifelse(train$balance <0, "Negative balance", "Zero balance"))))))
trainLR$balance_pos <- as.factor(trainLR$balance_pos)

testLR$balance_pos <- ifelse(test$balance > 0 & test$balance <= 100, "0 to 100",
                              ifelse(test$balance > 100 & test$balance <= 500, "100 to 500",
                                     ifelse(test$balance >500 & test$balance <= 2000, "500 to 2000",
                                            ifelse(test$balance >2000 & test$balance <= 10000, "2000 to 10000",
                                                   ifelse(test$balance >10000, "more than 10000",
                                                          ifelse(test$balance <0, "Negative balance", "Zero balance"))))))
testLR$balance_pos <- as.factor(testLR$balance_pos)

# Campaign: We will reduce to 4 levels
trainLR$campaign_cat <-  ifelse(trainLR$campaign == 1, "Campaign Level 1", ifelse(trainLR$campaign ==2, "Campaign Level 2", ifelse(trainLR$campaign==3, "Campaign Level 3", "Campaign Level 4")))
trainLR$campaign_cat <- as.factor(trainLR$campaign_cat)
testLR$campaign_cat <-  ifelse(testLR$campaign == 1, "Campaign Level 1", ifelse(testLR$campaign ==2, "Campaign Level 2", ifelse(testLR$campaign==3, "Campaign Level 3", "Campaign Level 4")))
testLR$campaign_cat <- as.factor(testLR$campaign_cat)

#pdays: add variable for Contacted/No contacted
trainLR$pdays_contacted <- ifelse(test$pdays != -1, 1, 0)
testLR$pdays_not_contacted <- ifelse(test$pdays == -1, 1, 0)

#Convert days to months and make it a categorical variable: 
trainLR$pmonths <- trainLR$pdays/30
trainLR$months_passed <- ifelse(trainLR$pmonths <0, "Not Contacted", ifelse(trainLR$pmonths >= 0 & trainLR$pmonths <=2, "1 or 2 months", ifelse(trainLR$pmonths >2 & trainLR$pmonths <=6, "3 to 6 months", "More than 6 months")))
trainLR$months_passed <- as.factor(trainLR$months_passed)
testLR$pmonths <- testLR$pdays/30
testLR$months_passed <- ifelse(testLR$pmonths <0, "Not Contacted", ifelse(testLR$pmonths >= 0 & testLR$pmonths <=2, "1 or 2 months", ifelse(testLR$pmonths >2 & testLR$pmonths <=6, "3 to 6 months", "More than 6 months")))
testLR$months_passed <- as.factor(testLR$months_passed)

#previous: we'll make this a categorical variable
trainLR$not_contacted <- ifelse(trainLR$previous == 0,1,0)
trainLR$contacted <- ifelse(trainLR$previous > 0 & trainLR$previous <=2, "1 or 2 previous contacts", ifelse(trainLR$previous >2, "More than 2 previous contacts", 0))
trainLR$contacted <- as.factor(trainLR$contacted)

testLR$not_contacted <- ifelse(testLR$previous == 0,1,0)
testLR$contacted <- ifelse(testLR$previous > 0 & testLR$previous <=2, "1 or 2 previous contacts", ifelse(testLR$previous >2, "More than 2 previous contacts", 0))
testLR$contacted <- as.factor(testLR$contacted)

#Drop unused variables
trainLR <- trainLR[,-c(1,6,10,13,14,15,24)]
testLR <- testLR[,-c(1,6,10,13,14,15,24)]

# Convert categorical to dummy variables
library(dummies)
trainLR <- dummy.data.frame(data = trainLR, names = c("job", "marital", "education","month", "poutcome","balance_pos", "campaign_cat", "months_passed", "contacted", "contact"))
trainLR$housing <- ifelse(trainLR$housing == "yes", 1, 0)
trainLR$loan <- ifelse(trainLR$loan == "yes", 1, 0)
trainLR$default <- ifelse(trainLR$default == "yes", 1, 0)
trainLR$y <- ifelse(trainLR$y == "yes", 1, 0)

testLR <- dummy.data.frame(data = testLR, names = c("job", "marital", "education","month", "poutcome","balance_pos", "campaign_cat", "months_passed", "contacted", "contact"))
testLR$housing <- ifelse(testLR$housing == "yes", 1, 0)
testLR$loan <- ifelse(testLR$loan == "yes", 1, 0)
testLR$default <- ifelse(testLR$default == "yes", 1, 0)
testLR$y <- ifelse(testLR$y == "yes", 1, 0)

#Use model selection from glmnet package
library(glmnet)
#Convert to matrices, as required by glmnet
xtrainLR <- data.matrix(trainLR[,c(1:42,44:66)])
ytrainLR <- data.matrix(trainLR[,43])

#Use cv fit to choose the model
cvLR <- cv.glmnet(xtrainLR, ytrainLR, family = "binomial", type.measure = "class", nlambda = 1000)

#Plot the CV fit lambda against number of parameters
plot(cvLR)

#Find out which parameters are included in the model with the lowest lambda
coef(cvLR, s = "lambda.min")

#Get the predictions for the logistic model
LRtrain.class <- predict(cvLR, newx = xtrainLR, type = "class")

#Compare the prediction to the real outcome
table(LRtrain.class, trainLR$y)

#Prediction on the test set
#Convert to matrices
xtestLR <- as.matrix(testLR[,c(1:42,44:66)])
ytestLR <- as.matrix(testLR[,43])

#Run model from training set on test set
LRtest.class <- predict(cvLR, newx = xtestLR, type = "class")
LRtest.prob <- predict(cvLR, newx = xtestLR, type = "response")

#Confusion matrix
table(LRtest.class, testLR$y)

#ROC curves
LRtest.pred <- prediction(LRtest.prob[,1], ytestLR)
roc.LRtest.perf = performance(LRtest.pred, measure = "tpr", x.measure = "fpr")
auc.test <- performance(LRtest.pred, measure = "auc")
auc.test <- auc.test@y.values
plot(roc.LRtest.perf, col = "darkorange2", lwd = 4, axes = FALSE, main = "ROC curve for Logistic Regression Model")
abline(a=0, b= 1, col = "dodgerblue4", lwd = 3)
text(x = .40, y = .6,paste("AUC = ", round(auc.test[[1]],4), sep = ""), col = "darkorange4")


#LDA model

#Extracting the continuous variables from the model
trainLDA <- train[,c(2,7,13,14,15,16,18)]

#Scaling the responses (z-score or Min-max normalization are available, I chose z-value)

trainLDA <- as.data.frame(scale(trainLDA[,-7]))
trainLDA$y <- train$y

#Creating the LDA model
library(MASS)
LDA <- lda(y ~ ., data = trainLDA)

# Predicting on the train set
predLDAtrain <- predict(LDA, trainLDA)
#Confusion matrix
table(predLDAtrain$class, trainLDA$y)

#Model testing
testLDA <- test[,c(2,7,13,14,15,16,18)]
testLDA <- as.data.frame(scale(testLDA[,-7]))
testLDA$y <- test$y
LDA.results <- predict( LDA, testLDA )

# Results - Confusion Matrix
library( caret )
t = table( LDA.results$class, test$y )
print( confusionMatrix( t ) )

#ROC curves
preds <-LDA.results$posterior
preds <- as.data.frame(preds)
pred <- prediction(preds[,2],testLDA$y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.test <- performance(pred, measure = "auc")
auc.test <- auc.test@y.values
plot(roc.perf, main = "ROC curve of LDA Model", col = "darkorange2", lwd = 4)
abline(a=0, b= 1, col = "dodgerblue4", lwd = 3)
text(x = .40, y = .6,paste("AUC = ", round(auc.test[[1]],3), sep = ""), col = "darkorange4")

# Implementing the random Forest

#Modify some variable names
library(randomForest)
trainRF <- fread("https://raw.githubusercontent.com/VolodymyrOrlov/MSDS6372_Project2/master/data/bank-balanced-train-d.csv")
trainRF$contacted <- ifelse(trainRF$pdays ==-1, 0,1)
trainRF$job_blue_collar <- trainRF$`job_blue-collar`
trainRF$job_self_employed <- trainRF$`job_self-employed`
trainRF <- trainRF[,-c(3,8)]
trainRF$y <- as.factor(ifelse(trainRF$y==1, "yes", "no"))

testRF <- fread("https://raw.githubusercontent.com/VolodymyrOrlov/MSDS6372_Project2/master/data/bank-test-d.csv")
testRF$contacted <- ifelse(testRF$pdays ==-1, 0,1)
testRF$job_blue_collar <- testRF$`job_blue-collar`
testRF$job_self_employed <- testRF$`job_self-employed`
testRF <- testRF[,-c(3,8)]
testRF$y <- as.factor(ifelse(testRF$y==1, "yes", "no"))

#Train the model
RF1 <- randomForest(y ~ ., data = trainRF, importance = TRUE)

#Confusion matrix
table(RF1$predicted, trainRF$y)

importance(RF1)
#Test the model
predRFtest <- predict(RF1, newdata = testRF)

#Confusion matrix
table(predRFtest, testRF$y)

#Create ROC curve
fit.pred<-predict(RF1,newdata=testRF,type="prob")
pred <- prediction(fit.pred[,2], testRF$y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.test <- performance(pred, measure = "auc")
auc.test <- auc.test@y.values
plot(roc.perf, main = "ROC curve for Random Forest Model", col = "darkorange3", lwd = r)
abline(a=0, b= 1, col = "dodgerblue4", lwd = 3)
text(x = .40, y = .6,paste("AUC = ", round(auc.test[[1]],3), sep = ""), col = "darkorange4")


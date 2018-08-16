---
title: "Objective 2 MD"
author: "Rene Pineda"
date: "August 16, 2018"
output:
  html_document:
    keep_md: true
---



### a. Logistic Regression Model with categorical variables


```r
library(dplyr)
library(data.table)
library(caret)
library(ROCR)

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
```

![](Objective_2_MD_files/figure-html/LRM-1.png)<!-- -->

```r
#Find out which parameters are included in the model with the lowest lambda
coef(cvLR, s = "lambda.min")
```

```
## 66 x 1 sparse Matrix of class "dgCMatrix"
##                                                   1
## (Intercept)                            -1.493309336
## jobadmin.                               0.136058325
## jobblue-collar                         -0.175171084
## jobentrepreneur                        -0.160931144
## jobhousemaid                           -0.419467139
## jobmanagement                           .          
## jobretired                              .          
## jobself-employed                       -0.016883990
## jobservices                             .          
## jobstudent                              0.421203503
## jobtechnician                           .          
## jobunemployed                           0.083008125
## jobunknown                              .          
## maritaldivorced                         .          
## maritalmarried                         -0.153586280
## maritalsingle                           .          
## educationprimary                       -0.106553691
## educationsecondary                      .          
## educationtertiary                       0.116824854
## educationunknown                        0.128193503
## default                                -0.007894184
## housing                                -0.595061618
## loan                                   -0.262193117
## contactcellular                         0.102891204
## contacttelephone                        .          
## contactunknown                         -1.084411099
## monthapr                                0.687466839
## monthaug                                .          
## monthdec                                0.599286188
## monthfeb                                0.396192399
## monthjan                               -0.403482088
## monthjul                               -0.256497187
## monthjun                                0.512558121
## monthmar                                2.238476386
## monthmay                               -0.132764901
## monthnov                               -0.323826326
## monthoct                                1.448878842
## monthsep                                1.655387389
## duration                                0.005221767
## poutcomefailure                         .          
## poutcomeother                           0.056472916
## poutcomesuccess                         2.063857655
## poutcomeunknown                        -0.242091476
## adult                                   .          
## middleaged                             -0.268732806
## elderly                                 0.843172407
## balance_pos0 to 100                    -0.135651009
## balance_pos100 to 500                   .          
## balance_pos2000 to 10000                0.076307416
## balance_pos500 to 2000                  .          
## balance_posmore than 10000              0.296748761
## balance_posNegative balance            -0.456146623
## balance_posZero balance                -0.200884884
## campaign_catCampaign Level 1            0.283459345
## campaign_catCampaign Level 2            .          
## campaign_catCampaign Level 3            .          
## campaign_catCampaign Level 4           -0.292345145
## pdays_contacted                         .          
## months_passed1 or 2 months              0.138634688
## months_passed3 to 6 months              0.535579563
## months_passedMore than 6 months         .          
## months_passedNot Contacted              .          
## not_contacted                           .          
## contacted0                              .          
## contacted1 or 2 previous contacts       .          
## contactedMore than 2 previous contacts  .
```

```r
#Get the predictions for the logistic model
LRtrain.class <- predict(cvLR, newx = xtrainLR, type = "class")

#Compare the prediction to the real outcome
table(LRtrain.class, trainLR$y)
```

```
##              
## LRtrain.class    0    1
##             0 2132  508
##             1  368 1992
```

```r
#Prediction on the test set
#Convert to matrices
xtestLR <- as.matrix(testLR[,c(1:42,44:66)])
ytestLR <- as.matrix(testLR[,43])

#Run model from training set on test set
LRtest.class <- predict(cvLR, newx = xtestLR, type = "class")
LRtest.prob <- predict(cvLR, newx = xtestLR, type = "response")

#Confusion matrix
table(LRtest.class, testLR$y)
```

```
##             
## LRtest.class   0   1
##            0 815  12
##            1 127  46
```

```r
#ROC curves
LRtest.pred <- prediction(LRtest.prob[,1], ytestLR)
roc.LRtest.perf = performance(LRtest.pred, measure = "tpr", x.measure = "fpr")
auc.test <- performance(LRtest.pred, measure = "auc")
auc.test <- auc.test@y.values
plot(roc.LRtest.perf, col = "darkorange2", lwd = 4, axes = FALSE, main = "ROC curve for Logistic Regression Model")
abline(a=0, b= 1, col = "dodgerblue4", lwd = 3)
text(x = .40, y = .6,paste("AUC = ", round(auc.test[[1]],4), sep = ""), col = "darkorange4")
```

![](Objective_2_MD_files/figure-html/LRM-2.png)<!-- -->

### B. Linear discriminant analysis

You can also embed plots, for example:


```r
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
```

```
##      
##         no  yes
##   no  2059  854
##   yes  441 1646
```

```r
#Model testing
testLDA <- test[,c(2,7,13,14,15,16,18)]
testLDA <- as.data.frame(scale(testLDA[,-7]))
testLDA$y <- test$y
LDA.results <- predict( LDA, testLDA )

# Results - Confusion Matrix
t = table( LDA.results$class, test$y )
print( confusionMatrix( t ) )
```

```
## Confusion Matrix and Statistics
## 
##      
##        no yes
##   no  599  10
##   yes 343  48
##                                           
##                Accuracy : 0.647           
##                  95% CI : (0.6165, 0.6767)
##     No Information Rate : 0.942           
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.1255          
##  Mcnemar's Test P-Value : <2e-16          
##                                           
##             Sensitivity : 0.6359          
##             Specificity : 0.8276          
##          Pos Pred Value : 0.9836          
##          Neg Pred Value : 0.1228          
##              Prevalence : 0.9420          
##          Detection Rate : 0.5990          
##    Detection Prevalence : 0.6090          
##       Balanced Accuracy : 0.7317          
##                                           
##        'Positive' Class : no              
## 
```

```r
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
```

![](Objective_2_MD_files/figure-html/LDA-1.png)<!-- -->

### c. Random Forest


```r
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
```

```
##      
##         no  yes
##   no  2055  284
##   yes  445 2216
```

```r
#Show which variables are important
importance(RF1)
```

```
##                               no         yes MeanDecreaseAccuracy
## day                  25.58306120   1.3358720           21.2181488
## job_admin.            1.95198873  -1.7291691            0.2064931
## job_entrepreneur     -2.10110780   0.7354142           -0.8865609
## job_housemaid         3.42027016   4.5886378            5.6227104
## job_management        3.29597591   3.8453999            5.3909589
## job_retired           4.24199287   6.1981125            8.2618537
## job_services         -0.34036718  -0.7162602           -0.7852295
## job_student           5.97240973   6.6908716            8.8073001
## job_technician        2.81016041   2.5603256            3.8968749
## job_unemployed       -0.34719983  -2.9841249           -2.4204593
## job_unknown          -2.01713976  -0.7358906           -1.7976299
## marital_divorced     -2.50726096   2.5197339            0.2395830
## marital_married       2.69676168   5.3463502            5.6518009
## marital_single        2.12349005   7.9163737            8.1632972
## education_primary     1.01535414   6.4738936            5.7623050
## education_secondary   2.40944792   1.4150766            2.8315812
## education_tertiary    3.61313346   4.7295186            6.4210399
## education_unknown     0.08488626   2.9886503            2.4550364
## default_no           -2.16074686   2.2847237            0.7142918
## default_yes          -0.55232251   4.7204980            3.2878485
## housing_no           17.18860713  18.1962438           21.5938872
## housing_yes          16.61625956  15.6545585           19.6176589
## loan_no               3.50318419  12.3985488           11.9023680
## loan_yes              2.26603548  12.4529999           11.3744828
## contact_cellular     18.86927632   9.3397210           20.6857324
## contact_telephone     1.94818885   7.0018315            6.4997030
## contact_unknown      29.17313981  13.2159058           30.6639871
## month_apr            24.50188680  21.7767146           30.6505303
## month_aug            28.29757372  -5.7104340           25.1332835
## month_dec             3.83685414  -1.2879661            1.9387099
## month_feb            19.97635373   6.9077499           20.8415226
## month_jan            16.60268406  -6.1422070           10.3026254
## month_jul            20.88116970   1.8431806           17.7173265
## month_jun            22.03118126   0.4382697           22.0554553
## month_mar            22.14762995  33.0109715           33.5461653
## month_may            18.57734675   4.7745502           19.9339702
## month_nov            21.86555569  -5.3378149           18.5414224
## month_oct            28.46377495  16.6544815           30.0316840
## month_sep            21.30920063   9.5297469           22.4914451
## poutcome_failure     14.92231481 -10.3161363            9.6430524
## poutcome_other        7.93537409  -4.1724510            4.0397636
## poutcome_success     15.80185989  33.9895101           30.4519373
## poutcome_unknown     12.52852817   8.4937915           13.2775744
## age                  20.58759105  20.6396775           28.4609496
## balance               9.55627306  13.1036305           16.1118995
## duration            122.45241671 137.7446419          148.4402341
## campaign             10.69250132   7.8502939           13.2834852
## pdays                22.17702491   9.1077653           21.2567276
## previous             12.10461204   8.8988626           12.7815006
## contacted            13.06600152   9.6701026           14.1510988
## job_blue_collar       2.03903939  10.9310930            9.8072304
## job_self_employed     1.32851904   0.4916013            1.2645898
##                     MeanDecreaseGini
## day                       129.104984
## job_admin.                 17.709994
## job_entrepreneur            8.242428
## job_housemaid               6.972551
## job_management             19.448753
## job_retired                11.519853
## job_services               12.879811
## job_student                 8.218188
## job_technician             20.122369
## job_unemployed              8.281665
## job_unknown                 1.333828
## marital_divorced           14.162814
## marital_married            23.098034
## marital_single             19.006713
## education_primary          14.706539
## education_secondary        21.732527
## education_tertiary         20.798446
## education_unknown           8.652097
## default_no                  3.213215
## default_yes                 3.136619
## housing_no                 39.494105
## housing_yes                37.989707
## loan_no                    16.796170
## loan_yes                   15.915858
## contact_cellular           38.457285
## contact_telephone           9.749311
## contact_unknown            59.600480
## month_apr                  33.825607
## month_aug                  26.301383
## month_dec                   2.922999
## month_feb                  21.101650
## month_jan                  12.800152
## month_jul                  23.668082
## month_jun                  22.313202
## month_mar                  29.435083
## month_may                  30.425182
## month_nov                  20.366627
## month_oct                  25.624611
## month_sep                  18.357756
## poutcome_failure           15.780927
## poutcome_other              7.048015
## poutcome_success           77.411404
## poutcome_unknown           27.864731
## age                       159.309783
## balance                   163.995350
## duration                  761.245770
## campaign                   72.800760
## pdays                      77.138857
## previous                   44.867588
## contacted                  28.340755
## job_blue_collar            22.039649
## job_self_employed           8.109480
```

```r
#Test the model
predRFtest <- predict(RF1, newdata = testRF)

#Confusion matrix
table(predRFtest, testRF$y)
```

```
##           
## predRFtest  no yes
##        no  778   8
##        yes 164  50
```

```r
#Create ROC curve
fit.pred<-predict(RF1,newdata=testRF,type="prob")
pred <- prediction(fit.pred[,2], testRF$y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.test <- performance(pred, measure = "auc")
auc.test <- auc.test@y.values
plot(roc.perf, main = "ROC curve for Random Forest Model", col = "darkorange3", lwd = 4)
abline(a=0, b= 1, col = "dodgerblue4", lwd = 3)
text(x = .40, y = .6,paste("AUC = ", round(auc.test[[1]],3), sep = ""), col = "darkorange4")
```

![](Objective_2_MD_files/figure-html/RF-1.png)<!-- -->

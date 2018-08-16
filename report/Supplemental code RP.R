# Duration: convert to categorical according to distribution
trainLR$duration <- trainLR$duration/60
trainLR$dshort <- ifelse(trainLR$duration <= 5, 1, 0)
trainLR$dmedium <- ifelse(trainLR$duration >5 & trainLR$duration <= 10, 1, 0)
trainLR$dlong <- ifelse(trainLR$duration > 10, 1, 0)
testLR$duration <- testLR$duration/60
testLR$dshort <- ifelse(testLR$duration <= 5, 1, 0)
testLR$dmedium <- ifelse(testLR$duration >5 & testLR$duration <= 10, 1, 0)
testLR$dlong <- ifelse(testLR$duration > 10, 1, 0)

trainRF2 <- trainLR
names(trainRF2) <- make.names(names(trainRF2))
trainRF2$y <- as.factor(ifelse(trainRF2$y==1, "yes", "no"))
trainRF2$duration <- trainRF2$duration/60
trainRF2$dshort <- ifelse(trainRF2$duration <= 5, 1, 0)
trainRF2$dmedium <- ifelse(trainRF2$duration >5 & trainRF2$duration <= 10, 1, 0)
trainRF2$dlong <- ifelse(trainRF2$duration > 10, 1, 0)

trainRF2 <- trainRF2[,-38]
RF2 <- randomForest(y ~ ., data = trainRF2, importance = TRUE, ntrees = 1000)

testRF2 <- testLR
names(testRF2) <- make.names(names(testRF2))
testRF2$y <- as.factor(ifelse(testRF2$y==1, "yes", "no"))
testRF2$duration <- testRF2$duration/60
testRF2$dshort <- ifelse(testRF2$duration <= 5, 1, 0)
testRF2$dmedium <- ifelse(testRF2$duration >5 & testRF2$duration <= 10, 1, 0)
testRF2$dlong <- ifelse(testRF2$duration > 10, 1, 0)
testRF2 <- testRF2[,-38]

predtestRF2 <- predict(RF2, newdata = testRF2)

pdf_document: default

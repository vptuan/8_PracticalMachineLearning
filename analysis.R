library(caret)
library(ggplot2)

# load data
training <- read.csv("./data/pml-training.csv",na.strings = c("NA",""))
head(training)
str(training)

# clean data
naRemovedTrain <- training[,colSums(is.na(training)) < 100]
tidyData <- subset(naRemovedTrain, colSums(is.na(training)) < 100, select = -c(X))
str(tidyData)
head(tidyData[,7:59])
gbmF1 <- train(classe ~.,data=tidyData[,7:59], method="gbm")

trellis.par.set(caretTheme())
ggplot(gbmF1)

fitControl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10)

gbmF2 <- train(classe ~.,data=tidyData[,7:59], method="gbm", trControl = fitControl)

bestPredict <- subset(tidyData, select = -c(1:6,10,11,21,23,26,35,51))
rfF1 <-  train(classe ~.,data=bestPredict, method="rf")


preProc <- preProcess((predictData[,-59]),method="pca",pcaComp=2)

modelF3 <- train(classe ~.,data=predictData, 
                 method="lda", preProcess = "pca")
# Predicting movement from mobile devices
MThayer  

## Synopsis

This project pretends to explore and build an accurate predictive model for the manner in which people did the exercise wearing devices.


##Loading and cleaning data

Here we will load the data and do some cleaning, such as managing NA's and looking to the distribution of the prefictors


```r
library(caret); library(dplyr); library(randomForest); library(gbm); library(lubridate); library(glmnet); library(stringr)

train<-read.csv("pml-training.csv", stringsAsFactors=F)#[1:8000, ]
test<-read.csv("pml-testing.csv", stringsAsFactors=F)
```

## Feature selection

We will explore the predictors and create new ones using the following steps:
 - Delete the X predictor because it's an ID
 - Convert to numerical all the columns that looks like numerical
 - Deal with NA values (converting to -1 or a number not in the scale of the predictor in the case it contains negatives)
 - Remove duplicated features. We found that max_yaw_belt and min_yaw_belt are equals; max_yaw_dumbbell and min_yaw_dumbbell are equals; and max_yaw_forearm and min_yaw_forearm are equals.
 - Remove features with zero variance


```r
### Removing the id
testX<-test$X
train$X=NULL; test$X<-NULL
train$cvtd_timestamp=NULL; test$cvtd_timestamp<-NULL

train[is.na(train)]<- -1
test[is.na(test)]<- -1

### Convert to numeric where it seemed to be a number
for(c in names(dplyr::select(train, -classe, -user_name))) {
   if(!is.numeric(train[,c])){
       #c="kurtosis_yaw_dumbbell"
       train[,c]=as.numeric(train[,c])
       test[,c]=as.numeric(test[,c])
       train[is.na(train[,c]),c]<-ifelse(is.na(min(train[,c])), 0, min(train[,c]))-1
       test[is.na(test[,c]),c]<-ifelse(is.na(min(train[,c])), 0, min(train[,c]))-1
   }
}

### The test and train set don't have the same columns. So, we use only the one that match
features<-intersect(names(train), names(test))
classe<-train$classe

dummies = dummyVars("~.",data=rbind(train[,features]), fullRank=T)
train=as.data.frame(predict(dummies, newdata=train))
test=as.data.frame(predict(dummies, newdata=test))

#Look for zero variance features and remove them
nsv<-nearZeroVar(train, saveMetrics=TRUE)

train<-train[,!nsv$zeroVar]
test<-test[,!nsv$zeroVar]

##### Removing identical features
features_pair <- combn(colnames(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[,f1] == train[,f2])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}
```

```
## max_yaw_belt and min_yaw_belt are equals.
## max_yaw_dumbbell and min_yaw_dumbbell are equals.
## max_yaw_forearm and min_yaw_forearm are equals.
```

```r
features <- setdiff(colnames(train), toRemove)
train<- train[,features]
test<- test[,features]
```

##Modeling

I have built the models using the features selected in the previous section. Our strategy here consisted on splitting our train set in 5-folds, and generating one prediction for each out of fold set. I tested several models such as random forest, lasso and caret, but finally I only used gbm (Generalized Boosting Models). I applied the Box Cox transformation and PCA with a threshold of 0.9. The predictions were saved in the form of probability for each one of the outcomes. In other words, the model gave 5 vectors corresponding to the estimated probabilities that the rows were classified as A, another vector with the probabilities of beign "B", and so on. 

In the next section I used those prediction probabilities to train a random forest model.


Lets see the code that creates the out of fold predictions using the gbm library

```r
set.seed(123)

### Predictions repository for train
n.preds <- 5 
preds <- matrix(0, nrow = nrow(train), ncol = n.preds)
predsTest <- matrix(0, nrow = nrow(test), ncol = n.preds)

### 5 Fold using createFolds
n.fold <- 5
folds<-createFolds(y=as.factor(classe), k=n.fold, list=TRUE, returnTrain = FALSE)


for (fold.id in 1:n.fold) {
    #fold.id=1
    train.id <- unlist(folds[-fold.id])
    valid.id <- unlist(folds[fold.id])
    
    #gbm model #####################################
    model.name="gbm"
    
    preProc<-preProcess(train[train.id,], method=c("BoxCox", "pca"), thresh = .9)
    trainPC<-predict(preProc, train[train.id,])
    validPC<-predict(preProc, train[valid.id,])
    testPC<-predict(preProc, test)
  
    trainPC$classe2<-classe[train.id]

    fitGbm<-gbm(as.factor(classe2) ~ ., data=trainPC, n.trees=220, interaction.depth=3, shrinkage=0.1, n.minobsinnode = 10, verbose=FALSE, distribution="multinomial")
    
    predC=predict(fitGbm, validPC, n.trees=220, type="response")
    predC2<- as.data.frame(predC)
    
    preds[valid.id, 1] <- predC2[,1]
    preds[valid.id, 2] <- predC2[,2]
    preds[valid.id, 3] <- predC2[,3]
    preds[valid.id, 4] <- predC2[,4]
    preds[valid.id, 5] <- predC2[,5]
    
    #Printing the accuracy on the validation set
    predGbm=apply(predC2, 1, which.max)
    cat("\n", model.name, sum(diag(table(as.numeric(as.factor(classe[valid.id])), as.numeric(predGbm))))/sum(table(as.numeric(as.factor(classe[valid.id])), as.numeric(predGbm))))
    
    #Predicting on the test set
    predC=predict(fitGbm, testPC, n.trees=220, type="response")
    predC2<- as.data.frame(predC)
    
    predsTest[, 1] <- predC2[,1]
    predsTest[, 2] <- predC2[,2]
    predsTest[, 3] <- predC2[,3]
    predsTest[, 4] <- predC2[,4]
    predsTest[, 5] <- predC2[,5]
    
}
```

```
## 
##  gbm 0.8428025
##  gbm 0.8430173
##  gbm 0.8545223
##  gbm 0.8397452
##  gbm 0.8447617
```

After the first model was trained, the prediction probabilities are added to the train set as follows:

```r
train$gbm1=preds[, 1]
train$gbm2=preds[, 2]
train$gbm3=preds[, 3]
train$gbm4=preds[, 4]
train$gbm5=preds[, 5]

test$gbm1=predsTest[, 1]/n.fold
test$gbm2=predsTest[, 2]/n.fold
test$gbm3=predsTest[, 3]/n.fold
test$gbm4=predsTest[, 4]/n.fold
test$gbm5=predsTest[, 5]/n.fold
```

Here a new model was fitted using the previous predictors along with the original predictors. PCA with a threshold of 0.8 is used to reduce the dimentionality of the original predictors.


```r
preProc<-preProcess(train[,features], method=c("BoxCox", "pca"), thresh = .8)
trainPC<-predict(preProc, train[,features])
testPC<-predict(preProc, test[,features])

trainFinal<-cbind(trainPC, dplyr::select(train, starts_with("gbm")))
testFinal<-cbind(testPC, dplyr::select(test, starts_with("gbm")))
names(trainFinal)
```

```
##  [1] "PC1"  "PC2"  "PC3"  "PC4"  "PC5"  "PC6"  "PC7"  "PC8"  "PC9"  "PC10"
## [11] "PC11" "PC12" "PC13" "PC14" "PC15" "PC16" "PC17" "PC18" "PC19" "PC20"
## [21] "PC21" "PC22" "PC23" "PC24" "PC25" "PC26" "PC27" "gbm1" "gbm2" "gbm3"
## [31] "gbm4" "gbm5"
```

```r
trainFinal$classe<-classe

combModFit<-train(as.factor(classe)~., method="rf", data=trainFinal,
             trControl = trainControl(method = "none"), verbose=FALSE, 
             tuneGrid=expand.grid(mtry=2))
```

Finaly, lets get the predictions for test set. 

```r
pred=predict(combModFit, newdata=testFinal)
predTrain=predict(combModFit)

cat("\n", "gam final", sum(diag(table(predTrain, classe)))/sum(table(predTrain, classe)))
```

```
## 
##  gam final 1
```

```r
#cat("\n", "gam final", sum(diag(table(pred, classeTest)))/sum(table(pred, classeTest)))

print(head(data.frame(X=testX, pred=pred), 20))
```

```
##     X pred
## 1   1    B
## 2   2    A
## 3   3    A
## 4   4    A
## 5   5    A
## 6   6    E
## 7   7    D
## 8   8    B
## 9   9    A
## 10 10    A
## 11 11    B
## 12 12    C
## 13 13    B
## 14 14    A
## 15 15    E
## 16 16    E
## 17 17    A
## 18 18    B
## 19 19    B
## 20 20    B
```


#LUIGI LEGITTIMO 21/10/2019 PARTE SPERIMENTALE TESI MAGISTRALE "INTELLIGENZA ARTIFICIALE, MACHINE LEARNING E DATA SCIENCE IN FINANZA"
#CREDIT APPROVAL 2017 source:
#https://www.kaggle.com/shravan3273/credit-approval
library(readr)
credit <- read_csv("C:/Users/Luigi Legittimo/Downloads/credit.csv")
View(credit)
credit1<-credit
#FATTORIZZO VARIABILI
credit1$default <- ifelse(credit1$default== 2, 1, 0) #1 default #0 no-default
credit1$default<-factor(credit1$default, levels= c(0, 1),labels= c(0,1))
credit1$credit_history[credit1$credit_history=="fully repaid"]<-"fully"
credit1$credit_history[credit1$credit_history=="fully repaid this bank"]<-"fully"
credit1$credit_history[credit1$credit_history=="delayed"]<-"delayed or repaid"
credit1$credit_history[credit1$credit_history=="repaid"]<-"delayed or repaid"
credit1$credit_history <- factor(credit1$credit_history, levels= c("fully","delayed or repaid", "critical" ), labels= c(1,2,3))
credit1$purpose[credit1$purpose=="car (used)"]<-"car (used), radio/tv or retraining"
credit1$purpose[credit1$purpose=="radio/tv"]<-"car (used), radio/tv or retraining"
credit1$purpose[credit1$purpose=="retraining"]<-"car (used), radio/tv or retraining"
credit1$purpose[credit1$purpose!="car (used), radio/tv or retraining"]<-"others"
credit1$purpose <- factor(credit1$purpose, levels= c("car (used), radio/tv or retraining", "others"), labels= c(1,2))
credit1$checking_balance[credit1$checking_balance=="1 - 200 DM"]<-"<0-200 DM"
credit1$checking_balance[credit1$checking_balance=="< 0 DM"]<-"<0-200 DM"
credit1$checking_balance[credit1$checking_balance!="<0-200 DM"]<-"sconosciuto"
credit1$checking_balance<- factor(credit1$checking_balance, levels= c("<0-200 DM", "sconosciuto"),labels= c(1,2))
credit1$savings_balance[credit1$savings_balance=="< 100 DM"]<-"0-500 DM"
credit1$savings_balance[credit1$savings_balance=="101 - 500 DM"]<-"0-500 DM"
credit1$savings_balance[credit1$savings_balance!="0-500 DM"]<-"501+ DM or unknown"
credit1$savings_balance<- factor(credit1$savings_balance, levels= c("0-500 DM", "501+ DM or unknown"),labels= c(1,2))
credit1$employment_length[credit1$employment_length=="> 7 yrs"]<-"4+ yrs"
credit1$employment_length[credit1$employment_length=="4 - 7 yrs"]<-"4+ yrs"
credit1$employment_length[credit1$employment_length=="0 - 1 yrs"]<-"0-4 yrs"
credit1$employment_length[credit1$employment_length=="1 - 4 yrs"]<-"0-4 yrs"
credit1$employment_length[credit1$employment_length=="unemployed"]<-"disoccupato"
credit1$employment_length<-factor(credit1$employment_length, levels= c("4+ yrs","0-4 yrs","disoccupato"),labels= c(1,2,3))
credit1$property[credit1$property!="real estate"]<-"others"
credit1$property<-factor(credit1$property, levels= c("real estate", "others"),labels= c(1,2))
credit1$installment_plan[credit1$installment_plan!="none"]<-"bank or stores"
credit1$installment_plan<-factor(credit1$installment_plan, levels= c("none","bank or stores"),labels= c(0,1))
credit1$installment_rate[credit1$installment_rate<3]<-"Low Rate"
credit1$installment_rate[credit1$installment_rate!="Low Rate"]<-"High Rate"
credit1$installment_rate<-factor(credit1$installment_rate, levels= c("Low Rate","High Rate"),labels= c(1,2))
credit1$housing[credit1$housing!="own"]<-"rent or for free"
credit1$existing_credits[credit1$existing_credits<2]<-"1"
credit1$existing_credits[credit1$existing_credits>1]<-"2-4"
credit1$existing_credits <- factor(credit1$existing_credits, levels= c("1","2-4" ),labels= c(1,2))
credit1$other_debtors[credit1$other_debtors!="guarantor"]<-"co-applicant or none"
credit1$other_debtors <- factor(credit1$other_debtors, levels= c("co-applicant or none", "guarantor"),labels= c(0,1))
credit1$job[credit1$job!="skilled employee"]<-"not skilled employee"
credit1$job<-factor(credit1$job, levels= c("skilled employee", "not skilled employee" ),labels= c(1,2))
#nuovo db senza varabili "residence history", "telephone" e "foreign worker".
credit1<-credit1[, -c(11,19,20)]
#NORMALIZZAZIONE
min.max <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}
credit1$months_loan_duration<-min.max(credit1$months_loan_duration)
credit1$amount<-min.max(credit1$amount)
credit1$age<-min.max(credit1$age)
#BILANCIAMENTO
df_default <- subset(credit1, credit1$default==1)
df_no.default <- subset(credit1, credit1$default==0)
balance_no.default = sort(sample(nrow(df_no.default), 300))
df_no.default<-df_no.default[balance_no.default,]
credit.bilanciato = rbind(df_default, df_no.default)
table(credit.bilanciato$default)
#SPLIT training & test set
split = sort(sample(nrow(credit.bilanciato), nrow(credit.bilanciato)*.7))
train<-credit.bilanciato[split,]
test<-credit.bilanciato[-split,] 
library(caret)
#APPLICAZIONE ALGORITMI:
#REGRESSIONE LOGISTICA
glm.model <- glm(default ~.,family=binomial(link='logit'),data=train)
glm.pred.train = predict(glm.model,train, type = "response")
df.glm.train = cbind(train, glm.pred.train)
df.glm.train$previsione <- as.factor(ifelse(df.glm.train$glm.pred.train>0.5, 1, 0))
confusionMatrix(df.glm.train$previsione,df.glm.train$default, positive = '1')
glm.pred.test = predict(glm.model,test, type = "response")
df.glm.pred.test = cbind(test, glm.pred.test)
df.glm.pred.test$previsione <- as.factor(ifelse(df.glm.pred.test$glm.pred.test>0.5, 1, 0))
confusionMatrix(df.glm.pred.test$prevision,df.glm.pred.test$default, positive = '1')
#SUPPORT VECTOR MACHINE
library(e1071)
svm.model <- svm(default ~., train, method = "C-classification", kernel = "radial", cost = 10, gamma = 0.1)
svm.pred <- predict(svm.model, test)
table(svm.pred, test$default)
df.svm<-cbind(test, svm.pred)
df.svm$previsione <- as.factor(ifelse(df.svm$svm.pred==0, 0, 1))
confusionMatrix(test$default, df.svm$previsione, positive = '1')
#KNN (solo con variabili quantitative)
library(class)
knn.model <- knn(train = train[,c(2,5,7,8,12,15), drop=F], test = test[,c(2,5,7,8,12,15), drop=F], cl = train[,16, drop=T], k = 3)
class.knn.test<-test[,16]
df.knn<-cbind(knn.model, class.knn.test)
confusionMatrix(df.knn$knn.model, df.knn$default, positive = '1')
#NAIVE BAYES #package: e1071
naivebayes.model <- naiveBayes(default ~., data = train)
naivebayes.pred <- predict(naivebayes.model, test)
confusionMatrix(table(naivebayes.pred, test$default), positive = '1')
#ENSAMBLE: BOOSTING 
library(ada)
boosting.model <- ada(default ~., train, loss = "logistic")
boosting.pred <- predict(boosting.model, test)
confusionMatrix(table(boosting.pred, test$default), positive = '1')
#RANDOM FOREST (va senza le due variabili qualitatitive)
library(randomForest)
randomforest.model <- randomForest(default ~ checking_balance + months_loan_duration + credit_history + 
                                     amount + savings_balance + employment_length + installment_rate + 
                                     property + age + installment_plan + existing_credits, data = train)
importance(randomforest.model)
varImpPlot(rf)
randomforest.pred<-predict(randomeforest.model, test, type = "class")
confusionMatrix(table(randomforest.model, test$default), positive = '1')
#DECISION TREE (CART)
library(rpart)
alberodecisione.model <- rpart(default ~., train)
alberodecisione.pred <- predict(alberodecisione.model, test, type = "class")
library(rattle)
fancyRpartPlot(alberodecisione.model, tweak=1.3)
confusionMatrix(table(alberodecisione.pred, test$default), positive = '1')
#ARTIFICIAL NEURAL NETWORK
library(nnet)
NeuralNet.model <- nnet(default ~., train, size = 10, rang = 0.1, maxit = 200)
NeuralNet.pred <- predict(NeuralNet.model, test, type = "class")
confusionMatrix(table(test$default, NeuralNet.pred),positive = '1')
library(NeuralNetTools)
plotnet(NeuralNet.model)
library(ggplot2)
garson(NeuralNet.model) + coord_flip()
#COMPARARE ACCURATEZZA,SENSITIVITY ECC. 
#FINE
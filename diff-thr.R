setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
data<- read.csv("AxisBank1.csv",header = T)
data <- data[,c(3:8)]
str(data)

data <- data[, !apply(is.na(data),2,all)]
data <- data[!apply(is.na(data),1,all),]
str(data)
head(data)
newdata = data


apply(data, 2 ,range)


max <- unname(apply(data,2,max,na.rm=T))
min <- unname(apply(data,2,min,na.rm=T))


##data transformation
data<- as.data.frame(scale(data, center= min, scale = max-min))


## Partitioning data 70%:30%
partidx<- sample(nrow(data)*0.7,replace = F)
train <- data[partidx,]
test <- data[-partidx,]

allVars<- colnames(data)
predictorVars <- allVars[!allVars%in%"Close.Price"]
predictorVars <- paste(predictorVars,collapse = "+")


## Training neural net
form= as.formula(paste("Close.Price~",predictorVars,collapse = "+"))
epoch = nrow(train1); epoch
library(neuralnet)
library(rminer)

## Using rep=20 and selecting the best model using validation partition
th=seq(0.001,0.1,0.005);th
Mtest=NULL  # to storevalidation error value
for (i in th) {
  mod=neuralnet(form,algorithm = "rprop+", threshold = i, stepmax = 50*epoch,
                 data=train, hidden = c(5,2), linear.output = T,
                 rep = 20)
  best=as.integer(which.min(mod$result.matrix[c("error"),]))
  modtest=compute(mod, test[,-c(5)],rep = best)
  M=mmetric(test$Close.Price,modtest$net.result[,1],c("RMSE"))
  Mtest=c(Mtest,M)
  
}
DF=data.frame("threshold"=th,"error"=Mtest);DF
DF[which.min(DF$error),]
plot(th,Mtest,type = "b",
     xlab = "threshold", ylab = "Validation Error",col="red")




## bEST MOdel
mod5=neuralnet(form,algorithm = "rprop+", threshold = 0.001, stepmax = 100*epoch,
               data = train, hidden = c(5,2), linear.output = T,
               rep=20)
best=as.integer(which.min(mod5$result.matrix[c("error"),]))
mod5$result.matrix[1:3,best]
M7=mmetric(train$Close.Price,mod5$net.result[[best]][,1],c("RMSE"));M7

mod5test=compute(mod5,test[,-c(3)], rep=best)
M8=mmetric(test$Close.Price,mod5test$net.result[,1],c("RMSE")); M8


## Scalling back (numeric) outcome variable to original units
b=max[3];b
a=min[3];a

test.pred.org=a+(b-a)*mod5test$net.result[,1]
data.frame("Actual Value"= newdata[-partidx,5],"Predicted Value"=test.pred.org)


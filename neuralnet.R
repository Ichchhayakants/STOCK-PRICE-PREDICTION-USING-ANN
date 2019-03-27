setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
data<- read.csv("AxisBank1.csv",header = T)
data <- data[,-c(1,2)]
str(data)

data <- data[, !apply(is.na(data),2,all)]
data <- data[!apply(is.na(data),1,all),]
str(data)
head(data)
newdata = data

## Variable transformation and normalization
max=unname(apply(data,2,max,na.rm=T))
min=unname(apply(data,2,min,na.rm=T))
data=as.data.frame(scale(data,center = min,
                     scale = max-min))

# partitioning data 90%:10%
partidx= sample(1:nrow(data), 0.7*nrow(data), replace = F)
train = data[partidx,]
test = data[-partidx,]
library(neuralnet)

# input layer: 11
# Hidden layer : 9:17
# output layer : 1

mf = as.formula(paste("Close.Price ~", paste(names(data)[!names(data)%in% "Close.Price"],
                                       collapse = "+")))
epoch = nrow(train); epoch
mod1 = neuralnet(mf,threshold = 0.01, stepmax = 50*epoch,
                 data = train, hidden = c(9), linear.output = T, rep= 1) 

mod2 = neuralnet(mf, algorithm = "rprop+",threshold = 0.04, stepmax = 2*epoch,
                 data = train, hidden = c(9), linear.output = T, rep= 1) 
mod1$result.matrix[1:3,1]
mod2$result.matrix[1:3,1]

neuralModel <- neuralnet(mf,algorithm = "rprop+", threshold = 0.001,stepmax = 30*epoch,
                        data=train,hidden=c(9,5),linear.output=T, rep = 1)


neuralModel$result.matrix1[1:3,1]
neuralModel$result.matrix[1:3,]

plot(neuralModel)

#interlayer connection weights
# input layer to hidden layer connections
dimnames(mod1$weights[[1]][[1]])= list(c("bias","node1:Prev.Close","node2:Open.Price",
                                         "node3:High.Price","node4:Low.Price",
                                         "node5:Average.Price","node6:Total.Trade","node7:MA30",
                                         "node8:MA50","node9:MA100","node10:MA200"),
                                       c("node13","node14","node15",
                                         "node16","node17","node18",
                                         "node19","node20","node21"))
mod1$weights[[1]][[1]]

dimnames(mod1$weights[[1]][[2]])= list(c("bias","node9","node10","node11",
                                         "node12","node13","node14",
                                         "node15","node16","node17"),
                                       c("node18:Price")); mod1$weights[[1]][[2]]
head(data.frame("predicted value"=mod1$net.result[[1]][[1]],
                "actual value"= train$Close.Price, train))

# performance
library(rminer)
## Training partition
M= mmetric(train$Close.Price, mod1$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M, digits = 6), na.print = "")

M1= mmetric(train$Close.Price, mod2$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M1,digits = 6),na.print = "")

#test aprtition
mod1test=compute(mod1, test[,-c(5)])
m2=mmetric(test$Close.Price, mod1test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m2,digits = 6),na.print = "")

mod2test=compute(mod2,test[,-c(5)])
m3=mmetric(test$Close.Price, mod2test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m3,digits = 6),na.print = "")

# network diagram
plot(mod1)


## network with 18 hidden nodes
mod3 = neuralnet(mf, algorithm = "rprop+",threshold = 0.07, stepmax = 35*epoch,
                 data= train, hidden = c(18), linear.output = T, rep = 1)
mod3$result.matrix[1:3,1]

# performance
# training partition
M4= mmetric(train$Close.Price,mod3$net.result[[1]][[1]],c("SSE","RMSE","ME"))
print(round(M4, digits = 6), na.print = "")

# test partition
mod3test = compute(mod3, test[,-c(5)])
M5=mmetric(test$Close.Price, mod3test$net.result[,1],c("SSE","RMSE","ME"))
print(round(M5,digits = 6), na.print = "")





## Using rep=20 and selecting the best model using validation partition
th=seq(0.01,0.1,0.005);th
Mtest=NULL  # to storevalidation error value
for (i in th) {
  mod4=neuralnet(mf,algorithm = "rprop+", threshold = i, stepmax = 35*epoch,
                 data=train, hidden = c(9), linear.output = T,
                 rep = 20)
  best=as.integer(which.min(mod4$result.matrix[c("error"),]))
  mod4test=compute(mod4, test[,-c(3)],rep = best)
  M6=mmetric(test$Close.Price,mod4test$net.result[,1],c("RMSE"))
  Mtest=c(Mtest,M6)
  
}

DF=data.frame("threshold"=th,"error"=Mtest);DF
DF[which.min(DF$error),]
plot(th,Mtest,type = "b",
     xlab = "threshold", ylab = "Validation Error")

## bEST MOdel
mod5=neuralnet(mf,algorithm = "rprop+", threshold = 0.08, stepmax = 30*epoch,
               data = train, hidden = c(9), linear.output = T,
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


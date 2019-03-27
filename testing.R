setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
data1<- read.csv("AxisBank1.csv",header = T)
data2<- read.csv("10year.csv",header = T)
data3<- read.csv("1Year.csv",header = T)
data1 <- data1[,-c(1,2)]
data2 <- data2[,-c(1,2,3)]
data3 <- data3[,-c(1,2,3)]
str(data1)
str(data2)
str(data3)

data2$Total.Traded.Quantity=as.numeric(data2$Total.Traded.Quantity)
data3$Total.Traded.Quantity=as.numeric(data3$Total.Traded.Quantity)

data1 <- data1[, !apply(is.na(data1),2,all)]
data1 <- data1[!apply(is.na(data1),1,all),]

data2 <- data2[, !apply(is.na(data2),2,all)]
data2 <- data2[!apply(is.na(data2),1,all),]

data3 <- data3[, !apply(is.na(data3),2,all)]
data3 <- data3[!apply(is.na(data3),1,all),]

newdata1 = data1
newdata2 = data2
newdata3 = data3

apply(data1, 2 ,range)
apply(data2, 2 ,range)
apply(data3, 2 ,range)

max1 <- unname(apply(data1,2,max,na.rm=T))
min1 <- unname(apply(data1,2,min,na.rm=T))

max2 <- unname(apply(data2,2,max,na.rm=T))
min2 <- unname(apply(data2,2,min,na.rm=T))

max3 <- unname(apply(data3,2,max,na.rm=T))
min3 <- unname(apply(data3,2,min,na.rm=T))


str(data2)
str(data1)
##data transformation
data1<- as.data.frame(scale(data1, center= min1, scale = max1-min1))
data2<- as.data.frame(scale(data2, center= min2, scale = max2-min2))
data3<- as.data.frame(scale(data3, center= min3, scale = max3-min3))

## Partitioning data 70%:30%
partidx1<- sample(nrow(data1)*0.7,replace = F)
partidx2<- sample(nrow(data2)*0.7,replace = F)
partidx3<- sample(nrow(data3)*0.7,replace = F)
train1 <- data1[partidx1,]
train2<- data2[partidx2,]
train3 <- data3[partidx3,]

test1 <- data1[-partidx1,]
test2 <- data2[-partidx2,]
test3 <- data3[-partidx3,]

allVars1<- colnames(data1)
predictorVars1 <- allVars1[!allVars1%in%"Close.Price"]
predictorVars1 <- paste(predictorVars1,collapse = "+")

allVars2<- colnames(data2)
predictorVars2 <- allVars2[!allVars2%in%"Close.Price"]
predictorVars2 <- paste(predictorVars2,collapse = "+")

allVars3<- colnames(data3)
predictorVars3 <- allVars3[!allVars3%in%"Close.Price"]
predictorVars3 <- paste(predictorVars3,collapse = "+")



## Training neural net
form1= as.formula(paste("Close.Price~",predictorVars1,collapse = "+"))
epoch1 = nrow(train1); epoch1

form2= as.formula(paste("Close.Price~",predictorVars2,collapse = "+"))
epoch2 = nrow(train2); epoch2

form3= as.formula(paste("Close.Price~",predictorVars3,collapse = "+"))
epoch3 = nrow(train3); epoch3


#### Model training
library(neuralnet)

mod1 = neuralnet(formula = form1,data=train1,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch1,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 

mod2 = neuralnet(formula = form2,data=train2,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch2,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 

mod3 = neuralnet(formula = form3,data=train3,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch3,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 


neuralnet <- function (formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, 
                       rep = 1, startweights = NULL, learningrate.limit = NULL, 
                       learningrate.factor = list(minus = 0.5, plus = 1.2), learningrate = NULL, 
                       lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", 
                       err.fct = "sse", act.fct = "logistic", linear.output = TRUE, 
                       exclude = NULL, constant.weights = NULL, likelihood = FALSE) 
{
  call <- match.call()
  options(scipen = 100, digits = 10)
  result <- varify.variables(data, formula, startweights, learningrate.limit, 
                             learningrate.factor, learningrate, lifesign, algorithm, 
                             threshold, lifesign.step, hidden, rep, stepmax, err.fct, 
                             act.fct)
  data <- result$data
  formula <- result$formula
  startweights <- result$startweights
  learningrate.limit <- result$learningrate.limit
  learningrate.factor <- result$learningrate.factor
  learningrate.bp <- result$learningrate.bp
  lifesign <- result$lifesign
  algorithm <- result$algorithm
  threshold <- result$threshold
  lifesign.step <- result$lifesign.step
  hidden <- result$hidden
  rep <- result$rep
  stepmax <- result$stepmax
  model.list <- result$model.list
  matrix <- NULL
  list.result <- NULL
  result <- generate.initial.variables(data, model.list, hidden, 
                                       act.fct, err.fct, algorithm, linear.output, formula)
  covariate <- result$covariate
  response <- result$response
  err.fct <- result$err.fct
  err.deriv.fct <- result$err.deriv.fct
  act.fct <- result$act.fct
  act.deriv.fct <- result$act.deriv.fct
  for (i in 1:rep) {
    if (lifesign != "none") {
      lifesign <- display(hidden, threshold, rep, i, lifesign)
    }
    utils::flush.console()
    result <- calculate.neuralnet(learningrate.limit = learningrate.limit, 
                                  learningrate.factor = learningrate.factor, covariate = covariate, 
                                  response = response, data = data, model.list = model.list, 
                                  threshold = threshold, lifesign.step = lifesign.step, 
                                  stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                                  startweights = startweights, algorithm = algorithm, 
                                  err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                                  act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                                  rep = i, linear.output = linear.output, exclude = exclude, 
                                  constant.weights = constant.weights, likelihood = likelihood, 
                                  learningrate.bp = learningrate.bp)
    if (!is.null(result$output.vector)) {
      list.result <- c(list.result, list(result))
      matrix <- cbind(matrix, result$output.vector)
    }
  }
  utils::flush.console()
  if (!is.null(matrix)) {
    weight.count <- length(unlist(list.result[[1]]$weights)) - 
      length(exclude) + length(constant.weights) - sum(constant.weights == 
                                                         0)
    if (!is.null(startweights) && length(startweights) < 
        (rep * weight.count)) {
      warning("some weights were randomly generated, because 'startweights' did not contain enough values", 
              call. = F)
    }
    ncol.matrix <- ncol(matrix)
  }
  else ncol.matrix <- 0
  if (ncol.matrix < rep) 
    warning(sprintf("algorithm did not converge in %s of %s repetition(s) within the stepmax", 
                    (rep - ncol.matrix), rep), call. = FALSE)
  nn <- generate.output(covariate, call, rep, threshold, matrix, 
                        startweights, model.list, response, err.fct, act.fct, 
                        data, list.result, linear.output, exclude)
  return(nn)
}
<environment: namespace:neuralnet>
  

mod1$result.matrix[1:3,1]  # 17 years
mod2$result.matrix[1:3,1]  # 10 years
mod3$result.matrix[1:3,1]  # 1 years

plot(mod1)
plot(mod2)
plot(mod3)

#interlayer connection weights
# input layer to hidden layer connections
dimnames(mod1$weights[[1]][[1]])= list(c("bias","node1:Prev.Close","node2:Open.Price",
                                         "node3:High.Price","node4:Low.Price",
                                         "node5:Average.Price","node6:Total.Trade","node7:MA30",
                                         "node8:MA50","node9:MA100","node10:MA200"),
                                       c("node11","node12","node13",
                                         "node14","node15","node16",
                                         "node17","node18","node19"))
mod1$weights[[1]][[1]]

dimnames(mod1$weights[[1]][[2]])= list(c("bias","node11","node12",
                                         "node13","node14","node15",
                                         "node16","node17","node18","node19"),
                                       c("node20","node21","node22",
                                         "node23","node24"))
mod1$weights[[1]][[2]]
head(data.frame("predicted value"=mod1$net.result[[1]][[1]],
                "actual value"= train1$Close.Price, train1))

# performance
library(rminer)
## Training partition
M1= mmetric(train1$Close.Price, mod1$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M1, digits = 6), na.print = "")

M2= mmetric(train2$Close.Price, mod2$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M2,digits = 6),na.print = "")

M3= mmetric(train3$Close.Price, mod3$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M3,digits = 6),na.print = "")

#test partition
mod1test=compute(mod1, test1[,-c(5)])
m1=mmetric(test1$Close.Price, mod1test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m1,digits = 6),na.print = "")

mod2test=compute(mod2,test2[,-c(5)])
m2=mmetric(test2$Close.Price, mod2test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m2,digits = 6),na.print = "")

mod3test=compute(mod3,test3[,-c(5)])
m3=mmetric(test3$Close.Price, mod3test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m3,digits = 6),na.print = "")







## Using rep=20 and selecting the best model using validation partition
th=seq(0.001,0.1,0.005);th
Mtest=NULL  # to storevalidation error value
for (i in th) {
  mod4=neuralnet(form,algorithm = "rprop+", threshold = i, stepmax = 50*epoch,
                 data=train, hidden = c(9,5), linear.output = T,
                 rep = 20)
  best=as.integer(which.min(mod4$result.matrix[c("error"),]))
  mod4test=compute(mod4, test[,-c(5)],rep = best)
  M6=mmetric(test$Close.Price,mod4test$net.result[,1],c("RMSE"))
  Mtest=c(Mtest,M6)
  
}
str(test)
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







test1.pred.org=min1+(max1-min1)*mod1test$net.result[,1]
pred1 <- data.frame("Actual Value"= newdata1[-partidx1,5],"Predicted Value"=test1.pred.org)

test2.pred.org=min2+(max2-min2)*mod2test$net.result[,1]
pred2<- data.frame("Actual Value"= newdata2[-partidx2,5],"Predicted Value"=test2.pred.org)

test3.pred.org=min3+(max3-min3)*mod3test$net.result[,1]
pred3<- data.frame("Actual Value"= newdata3[-partidx3,5],"Predicted Value"=test3.pred.org)

library(dplyr)
pred1 <-filter(pred1,pred1$Predicted.Value<=1600)
pred2 <- filter(pred2,pred2$Predicted.Value<=1600)
pred3 <- filter(pred3,pred3$Predicted.Value<=1600)





library(ggplot2)

attach(pred1)
par(las=1)
plot(Actual.Value,Predicted.Value,type = "n",main = "Actual v/s Predicted price",
     xlab = "time",ylab = "price")
points(Actual.Value,pch='+')
lines(Actual.Value,col="red",lty=1)
points(Predicted.Value,pch='*',col="blue")
lines(Predicted.Value,col="green",lty=2)
legend(0.8,9.25,c("Actual price","Predicted Price for complete data"),lty = c(1,2))
detach(pred1)

attach(pred2)
par(las=1)
plot(Actual.Value,Predicted.Value,type = "n",main = "Actual v/s Predicted pricefor 10 years data",
     xlab = "time",ylab = "price")
points(Actual.Value,pch='+')
lines(Actual.Value,col="red",lty=1)
points(Predicted.Value,pch='*',col="blue")
lines(Predicted.Value,col="green",lty=2)
legend(0.8,9.25,c("Actual price","Predicted Price"),lty =c(1,2))
detach(pred2)


attach(pred3)
par(las=1)
plot(Actual.Value,Predicted.Value,type = "n",main = "Actual v/s Predicted price",
     xlab = "time",ylab = "price")
points(Actual.Value)
lines(Actual.Value,col="black")
lines(Predicted.Value,col="green")
legend(0.8,9.25,c("Actual price","Predicted Price"),lty = c(1,2))
detach(pred3)

attach(pred3)
par(las=1)
plot(Actual.Value,Predicted.Value,type = "n")
plot(Predicted.Value,type = "l",col="red")
plot(Actual.Value,type = "l",col="blue")
detach(pred3)


lines(pred3$Actual.Value)
lines(pred3$Predicted.Value,col="green")



library(reshape2)
pred1 <- melt(pred1,id.vars ="Actual.Value")
ggplot(pred1,aes(Actual.Value,value,col=variable))+stat_smooth()+geom_line()

matplot(pred1$Actual.Value,pred1$Predicted.Value,type = "l")

library(tidyverse)
pred1 %>%tidyr::gather("id","value",1:2) %>%
ggplot(.,aes(pred1$Actual.Value,value))+geom_line()+geom_smooth(method="lm",se=F,color="blue")+
  facet_wrap(~id)

ggplot(pred1,aes(Actual.Value,Predicted.Value))+geom_line()  

str(pred1)  
  
str(pred1)
par(mfrow=c(1,2))
par(las=1)
plot(Actual.Value,Predicted.Value,type = "n",main = "Actual v/s Predicted price",
     xlab = "time",ylab = "price")
lines(Actual.Value,col="black")
lines(Predicted.Value,col="green")

npred1 <- max(pred1)
xrange <- range(pred1$Actual.Value)
yrange <- range(pred1$Predicted.Value)
plot(xrange,yrange)
colors <- rainbow(npred1)
linetype <- c(1:npred1)



lines(newdata1[-partidx1,5],test1.pred.org)
plot(test1$Close.Price)
plot(test1.pred.org)
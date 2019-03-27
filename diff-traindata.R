setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
data<- read.csv("AxisBank1.csv",header = T)
data <- data[,-c(1,2)]
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


## Partitioning data(different partitions)
partidx1<- sample(nrow(data)*0.9,replace = F)
partidx2<- sample(nrow(data)*0.8,replace = F)
partidx3<- sample(nrow(data)*0.7,replace = F)
partidx4<- sample(nrow(data)*0.6,replace = F)
partidx5<- sample(nrow(data)*0.5,replace = F)

train1 <- data[partidx1,]
train2<- data[partidx2,]
train3 <- data[partidx3,]
train4 <- data[partidx4,]
train5 <- data[partidx5,]

test1 <- data[-partidx1,]
test2 <- data[-partidx2,]
test3 <- data[-partidx3,]
test4 <- data[-partidx4,]
test5 <- data[-partidx5,]

allVars<- colnames(data)
predictorVars <- allVars[!allVars%in%"Close.Price"]
predictorVars <- paste(predictorVars,collapse = "+")


## Training neural net
form = as.formula(paste("Close.Price~",predictorVars,collapse = "+"))
epoch = nrow(train1); epoch
library(neuralnet)

mod1 = neuralnet(formula = form,data=train1,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 


mod2 = neuralnet(formula = form,data=train2,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 

mod3 = neuralnet(formula = form,data=train3,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 

mod4 = neuralnet(formula = form,data=train4,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 

mod5 = neuralnet(formula = form,data=train5,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 


mod1$result.matrix[1:3,1]
mod2$result.matrix[1:3,1]
mod3$result.matrix[1:3,1]
mod4$result.matrix[1:3,1]
mod5$result.matrix[1:3,1]

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
  
  
  ## Result options
names(neuralModel)
plot(mod1)
plot(mod2)
plot(mod3)
plot(mod4)
plot(mod5)


# performance
library(rminer)

## Training partition
M1= mmetric(train1$Close.Price, mod1$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M1, digits = 6), na.print = "")

M2= mmetric(train2$Close.Price, mod2$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M2,digits = 6),na.print = "")

M3= mmetric(train3$Close.Price, mod3$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M3,digits = 6),na.print = "")

M4= mmetric(train4$Close.Price, mod4$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M4,digits = 6),na.print = "")

M5= mmetric(train5$Close.Price, mod5$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M5,digits = 6),na.print = "")


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

mod4test=compute(mod4,test4[,-c(5)])
m4=mmetric(test4$Close.Price, mod4test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m4,digits = 6),na.print = "")

mod5test=compute(mod5,test5[,-c(5)])
m5=mmetric(test5$Close.Price, mod5test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m5,digits = 6),na.print = "")


Tdata1 <-c(m1)
Tdata2 <-c(m2)
newData <- rbind(Tdata1,Tdata2)

library(ggplot2)
dat <- data.frame(rbind(m1,m2,m3,m4,m5))
dat2<- data.frame(rbind(M1,M2,M3,M4,M5))
par(mfrow=c(3,2))
plot(dat$ME,type = "b",main = "train data")
plot(dat2$ME,type = "b",main = "test data")
plot(dat$RMSE,type = "b")
plot(dat2$RMSE,type = "b")
plot(dat$SSE,type = "b")
plot(dat2$SSE,type = "b")


par(mfrow=c(1,3))
plot(dat$SSE)
lines(dat$SSE)
plot(dat$RMSE)
lines(dat$RMSE)
plot(dat$ME)
lines(dat$ME)

attach(dat)
par(las=1)
plot(SSE,RMSE,type = "n",main = "Errors in different data sets",
     xlab = "Datasets",ylab = "Error")

points(SSE,pch="*",col="red")
lines(SSE)

M=mmetric(test1$Close.Price,modtest$net.result[,1],c("RMSE"))
Mtest=c(Mtest,M)

DF=data.frame("Traindata"=1,"error"=Mtest);DF
DF[which.min(DF$error),]
plot(DF$Traindata,Mtest,type = "b",
     xlab = "TrainData", ylab = "Validation Error",col="red")


## Scalling back (numeric) outcome variable to original units
b=max[3];b
a=min[3];a





test1.pred.org=a+(b-a)*mod1test$net.result[,1]
pred1 <- data.frame("Actual Value"= newdata[-partidx1,5],"Predicted Value"=test1.pred.org)

test2.pred.org=a+(b-a)*mod2test$net.result[,1]
pred2 <- data.frame("Actual Value"= newdata[-partidx2,5],"Predicted Value"=test2.pred.org)

test3.pred.org=a+(b-a)*mod3test$net.result[,1]
pred3 <- data.frame("Actual Value"= newdata[-partidx3,5],"Predicted Value"=test3.pred.org)


test4.pred.org=a+(b-a)*mod4test$net.result[,1]
pred4 <- data.frame("Actual Value"= newdata[-partidx4,5],"Predicted Value"=test4.pred.org)


test5.pred.org=a+(b-a)*mod5test$net.result[,1]
pred5 <- data.frame("Actual Value"= newdata[-partidx5,5],"Predicted Value"=test5.pred.org)



library(ggplot2)

par(mfrow=c(3,2))
plot(pred1$Actual.Value,pred1$Predicted.Value,type = "n",main = "Actual v/s Predicted price for k-fold-1",
     xlab = "time",ylab = "price")
#points(Actual.Value)
lines(pred1$Actual.Value,col="black")
lines(pred1$Predicted.Value,col="green")

plot(pred2$Actual.Value,pred2$Predicted.Value,type = "n",main = "Actual v/s Predicted price for k-fold-2",
     xlab = "time",ylab = "price")
#points(Actual.Value)
lines(pred2$Actual.Value,col="black")
lines(pred2$Predicted.Value,col="green")


plot(pred3$Actual.Value,pred3$Predicted.Value,type = "n",main = "Actual v/s Predicted price for k-fold-3",
     xlab = "time",ylab = "price")
#points(Actual.Value)
lines(pred3$Actual.Value,col="black")
lines(pred3$Predicted.Value,col="green")


plot(pred4$Actual.Value,pred4$Predicted.Value,type = "n",main = "Actual v/s Predicted price for k-fold-4",
     xlab = "time",ylab = "price")
#points(Actual.Value)
lines(pred4$Actual.Value,col="black")
lines(pred4$Predicted.Value,col="green")

plot(pred5$Actual.Value,pred5$Predicted.Value,type = "n",main = "Actual v/s Predicted price for k-fold-5",
     xlab = "time",ylab = "price")
#points(Actual.Value)
lines(pred5$Actual.Value,col="black")
lines(pred5$Predicted.Value,col="green")



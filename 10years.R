setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
data<- read.csv("10year.csv",header = T)
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


## Partitioning data 70%:30%
partidx1<- sample(nrow(data)*0.7,replace = F)
partidx2<- sample(nrow(data)*0.7,replace = F)
partidx3<- sample(nrow(data)*0.7,replace = F)
partidx4<- sample(nrow(data)*0.7,replace = F)
partidx5<- sample(nrow(data)*0.7,replace = F)

train <- data[partidx1,]
train2<- data[partidx2,]
train3 <- data[partidx3,]
train4 <- data[partidx4,]
train5 <- data[partidx5,]

test <- data[-partidx1,]
test2 <- data[-partidx2,]
test3 <- data[-partidx3,]
test4 <- data[-partidx4,]
test5 <- data[-partidx5,]

allVars<- colnames(data)
predictorVars <- allVars[!allVars%in%"Close.Price"]
predictorVars <- paste(predictorVars,collapse = "+")


## Training neural net
form= as.formula(paste("Close.Price~",predictorVars,collapse = "+"))
epoch = nrow(train); epoch
library(neuralnet)


mod1 = neuralnet(formula = form,data=train,hidden = c(9,5),threshold = 0.01, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 




mod2 = neuralnet(formula = form,data=train,hidden = c(9,5),threshold = 0.001, stepmax = 50*epoch,
                        rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 


mod1$result.matrix[1:3,1]
mod2$result.matrix[1:3,1]


neuralModel <- neuralnet(form, hidden=c(4,2),
                         linear.output=T, data = train)

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
  
  
  neuralModel
## Result options
names(neuralModel)

plot(neuralModel)


plot(mod1)





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
                "actual value"= train$Close.Price, train))

# performance
library(rminer)
## Training partition
M= mmetric(train$Close.Price, mod1$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M, digits = 6), na.print = "")

M1= mmetric(train$Close.Price, mod2$net.result[[1]][[1]], c("SSE","RMSE","ME"))
print(round(M1,digits = 6),na.print = "")

#test partition
mod1test=compute(mod1, test[,-c(5)])
m2=mmetric(test$Close.Price, mod1test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m2,digits = 6),na.print = "")

mod2test=compute(mod2,test[,-c(5)])
m3=mmetric(test$Close.Price, mod2test$net.result[,1],c("SSE","RMSE","ME"))
print(round(m3,digits = 6),na.print = "")

# network diagram
plot(mod1)


## network with 18 hidden nodes
mod3 = neuralnet(formula = form,data=train,hidden = c(9,5),threshold = 0.005, stepmax = 50*epoch,
                 rep = 1,algorithm = "rprop+",linear.output = TRUE,err.fct = "sse",act.fct = "logistic") 


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

test.pred.org=a+(b-a)*mod5test$net.result[,1]
data.frame("Actual Value"= newdata[-partidx,5],"Predicted Value"=test.pred.org)


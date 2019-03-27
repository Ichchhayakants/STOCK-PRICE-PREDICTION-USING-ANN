rm(list = ls())
.rs.restartR()

library(MASS)
library(neuralnet)

setwd("C:/Users/Ichchhayakant/Desktop/R/data/axis bank")
#### Setting the seed so the we get same result each time
## we run the neural net
data<- read.csv("AxisBank.csv", header = T)

str(data)
data <- data[,c(5,6,7,8,10,11,12)]



set.seed(1000)

hist(data$Close.Price)

dim(data)

head(data)


apply(data, 2 ,range)


maxValue <- apply(data,2,max)
minValue <- apply(data,2,min)

data<- as.data.frame(scale(data, center= minValue, scale = maxValue-minValue))

train <- sample(nrow(data)*0.7,replace = F)
traindata <- data[train,]
testdata <- data[-train,]

allVars<- colnames(data)
predictorVars <- allVars[!allVars%in%"Close.Price"]
predictorVars <- paste(predictorVars,collapse = "+")

form= as.formula(paste("Close.Price~",predictorVars,collapse = "+"))

neuralModel <- neuralnet(formula=form, hidden=c(4,2), 
                         linear.output=T, data = traindata)

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

neuralModel$result.matrix

out <- cbind(neuralModel$covariate,neuralModel$net.result[[1]])
head(out)
dimnames(out) <- list(NULL, 
                      c("Prev.Close", "Open.Price","High.Price",
                        "Low.Price","Average.Price","Total.Traded.Quantity","nn-output"))

head(neuralModel$generalized.weights[[1]])


str(testdata)

predictions <- compute(neuralModel,testdata[,-5])
str(predictions)
predictions$net.result

## predicting and unscalling
predictions <- predictions$net.result*(max(testdata$Close.Price)-min(testdata$Close.Price))+min(testdata$Close.Price)

actualValues <- (testdata$Close.Price)*(max(testdata$Close.Price)-min(testdata$Close.Price))+ min(testdata$Close.Price)


MSE <- sum((predictions-actualValues)^2)/nrow(testdata)
MSE


plot(testdata$Close.Price, predictions, col='blue', main = 'Real vs Predicted',pch=1, cex=0.9, type="p", xlab = "Actual", ylab= "Predicted")


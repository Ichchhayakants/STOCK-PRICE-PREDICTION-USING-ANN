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

neuralModel <- neuralnet(formula=form, hidden=c(4,2), linear.output=T, data = traindata)

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


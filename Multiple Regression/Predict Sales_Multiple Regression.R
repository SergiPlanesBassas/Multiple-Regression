####MULTIPLE REGRESSION_PREDICT SALES####

#Predict the sales in four different types of products (PC, Laptops, Netbooks, Smartphones)
#While assessing the effects service and customer reviews have on sales
#Predict sales for the four types in the newproducts

#### Load necessary packages ####
library(readr)
library(ggplot2)
library(lattice)
library(corrplot)
library(e1071)
library(randomForest)
library(caret)
library(dplyr)
library(tidyverse)
library(reshape)

####Load the Datasets####
exist_products <-read.csv("~/Desktop/Multiple Regression/existing_products.csv")
new_products <-read.csv("~/Desktop/Multiple Regression/existing_products.csv")
exist_products <-data.frame(exist_products)
new_products <-data.frame(new_products)

####Data exploration####
summary(exist_products)
names(exist_products)
summary(new_products)

str(exist_products)
str(new_products)

#Check for missing values (NA) and remove those observations
is.na(exist_products)
sum(is.na(exist_products))

#There are 15 observations that contain missing values (NA), all of them in the BestSellersRank column
colSums(is.na(exist_products))

#The bestsellers column will, thus, be removed completely
exist_products$BestSellersRank <- NULL
colSums(is.na(exist_products))


#Example how to finding Outlier
#boxplot(exist_products$Price)
#boxplot(exist_products$x4StarReviews)

#Check for outliers
boxplot(exist_products[2:ncol(exist_products)])

#Remove outliers from the dependent variable
outlier_Dataset <- exist_products
outlier_Column <- outlier_Dataset$Volume
outlier_Dataset <- outlier_Dataset[outlier_Column > (quantile(outlier_Column)[[2]] -
                                                       1.5*IQR(outlier_Column)),]
outlier_Dataset<- outlier_Dataset[outlier_Column < (quantile(outlier_Column)[[4]] +
                                                      1.5*IQR(outlier_Column)),]
exist_products <- outlier_Dataset

#Check that the outliers are cleared
boxplot(exist_products[2:ncol(exist_products)])

#Check for duplicated observations
sum(duplicated(exist_products[-which(names(exist_products) == 'ProductNum')]))
exist_products[!duplicated(exist_products[-which(names(exist_products) == 'ProductNum')]),]

sum(duplicated(new_products[-which(names(new_products) == 'ProductNum')]))
new_products[!duplicated(new_products[-which(names(new_products) == 'ProductNum')]),]

#Check for duplicates without the price
duplicated(exist_products[c('ProductType','Price')])

#There are duplicated values with different prices
#The duplicated values will be merged into one single value and the price will be the mean of the duplicated values
#Keep one of the duplicated observations (the first one, for example)
duplicatedproductsvalues <- exist_products[duplicated(exist_products[,-c(2,3)]),][1,]

#Calculate the mean of the price of the duplicated observations
mean_duplicated <- mean(exist_products$Price[duplicated(exist_products[-c(2,3)])])

#Remove the duplicated observations
exist_products <- exist_products[!duplicated(exist_products[-c(2,3)]),]

#Add one of the duplicated observations (the first one that had been saved previously)
exist_products<- rbind(exist_products,duplicatedproductsvalues)

#Change the price to the mean of the price of the duplicated values
exist_products[nrow(exist_products),'Price'] <- mean_duplicated

#Check for rows with missing reviews
nrow(exist_products[which(exist_products$x4StarReviews==0 & exist_products$x3StarReviews == 0 & exist_products$x2StarReviews == 0 & exist_products$x1StarReviews == 0),])

#There are 3 rows with missing x4,x3,x2,x1 reviews that will be removed
exist_products <- exist_products[-which(exist_products$x4StarReviews==0 & exist_products$x3StarReviews == 0 & exist_products$x2StarReviews == 0 & exist_products$x1StarReviews == 0),]

#Check the variables
ggplot(exist_products, aes(x=x4StarReviews, y = Volume)) +
  geom_point() +
  geom_smooth()
ggplot(exist_products, aes(x=x3StarReviews, y = Volume)) +
  geom_point() +
  geom_smooth()
ggplot(exist_products, aes(x=x2StarReviews, y = Volume)) +
  geom_point() +
  geom_smooth()
ggplot(exist_products, aes(x=x1StarReviews, y = Volume)) +
  geom_point() +
  geom_smooth()
ggplot(exist_products, aes(x=PositiveServiceReview, y = Volume)) +
  geom_point() +
  geom_smooth()
ggplot(exist_products, aes(x=NegativeServiceReview, y = Volume)) +
  geom_point() +
  geom_smooth()

#Removing the observations that the graphs before showed were distrustful
exist_products <- exist_products[-which(exist_products$ProductNum == 118),]
exist_products <- exist_products[-which(exist_products$ProductNum == 123),]
exist_products <- exist_products[-which(exist_products$ProductNum == 134),]
exist_products <- exist_products[-which(exist_products$ProductNum == 135),]

#Placing the dependent variable in the first position
exist_products <- subset(exist_products, select=c(ncol(exist_products),1:(ncol(exist_products)-1)))

#Check correlation
#Dummify with feature engineering
datadummy <- dummyVars(" ~ .", data = exist_products)
exist_products_dummy <- data.frame(predict(datadummy, newdata = exist_products))
corrdata <- cor(exist_products_dummy)

#Check correlation with corrplot
corrplot(corrdata)
round(corrdata, digits = 2)
corrplot(corrdata, method = "color", type = "lower",tl.srt = 45, tl.col = "black", order = "AOE", diag = F, cex.axis=0.3)
#corrplot(corrdata,tl.pos='n', method='pie')
#corrplot(corrdata, type='upper', method='pie')

#Remove attributes
#Look for high correlation with the dependent variable and remove attributes
corrdata_frame <- data.frame(corrdata)
corrdata_frame[lower.tri(corrdata_frame,diag=TRUE)] <- NA
correlations <- corrdata_frame %>% rownames_to_column("id") %>% gather(key = "key", value = "value", -id) %>% filter(value > 0.85 | value < -0.85)

#Plot the volume of sales by product, colored by product type
ggplot(exist_products, aes(x=reorder(ProductNum, -Volume), y = Volume, fill = ProductType)) + 
  geom_col(position = 'dodge') +
  theme(axis.text.x = element_text(angle = 90)) +
  xlab('Product Number') +
  ylab('Volume') +
  ggtitle('Volume of sales by product') +
  labs(fill = 'Product Type')

#Feature Engineering
exist_products$TotalServiceReviews <- exist_products$PositiveServiceReview + exist_products$NegativeServiceReview

#Dummify with feature engineering
datadummy <- dummyVars(" ~ .", data = exist_products)
exist_products_dummy <- data.frame(predict(datadummy, newdata = exist_products))

#Define training and testing sets
set.seed(107)
inTrain <- createDataPartition(y=exist_products_dummy$Volume,p=0.75,list=FALSE)
training <- exist_products_dummy[inTrain,]
testing <- exist_products_dummy[-inTrain,]
nrow(training)
nrow(testing)

#Modelling the training data and testing with the test data
#Set the model training method to a 1 time repeated cross validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#Train the model
a <- c("Volume ~ x4StarReviews", "Volume ~ x4StarReviews + PositiveServiceReview", "Volume ~ x4StarReviews + TotalServiceReviews", "Volume ~ PositiveServiceReview")
b <- c("knn", "rf","svmLinear")
compare_var_mod <- c()
compare_models <- c()

for ( i in a) {
  for (j in b) {
    model <- train(formula(i), data = training, method = j, trControl = ctrl, preProcess = c('center','scale'), tuneLength = 20)
    pred <- predict(model, newdata = testing)
    pred_metric <- postResample(testing$Volume, pred)
    compare_models <- c(compare_models,model)
    compare_var_mod <- cbind(compare_var_mod , pred_metric)
  }
}
names_var <- c()
for (i in a) {
  for(j in b) {
    names_var <- append(names_var,paste(i,j))
  }
}
colnames(compare_var_mod) <- names_var
compare_var_mod

#Melt the error metrics
compare_var_mod_melt <- melt(compare_var_mod, varnames=c("metric","model"))
compare_var_mod_melt <- as.data.frame(compare_var_mod_melt)
compare_var_mod_melt

#Plot the error metrics
ggplot(compare_var_mod_melt, aes(x=model, y=value, fill = model)) +
  geom_col() +
  facet_grid(metric~., scales="free") +
  theme(axis.text.x = element_blank(), axis.ticks = element_blank()) +
  xlab('') +
  ggtitle('Error metrics comparison') +
  labs(fill = 'Features and model used')

#Predict for the new products
#Retrain and retest the model using the best features and algorithm
model <- train(Volume ~ x4StarReviews + TotalServiceReviews, data = training, method = 'rf', trControl = ctrl, preProcess = c('center','scale'), tuneLength=20)
model
pred <- predict(model, newdata = testing)
pred_metric <- postResample(testing$Volume, pred)
pred_metric
error <- data.frame(pred - testing$Volume)
colnames(error) <- c("err")
ggplot(error, aes(err)) + 
  geom_histogram(bins=20)

#Make predictions for the new attributes
new_products_undummy <- new_products
new_datadummy <- dummyVars(" ~ .", data = new_products)
new_products <- data.frame(predict(new_datadummy, newdata = new_products))

#Placing the dependent variable in the first position
new_products <- subset(new_products, select=c(ncol(new_products),1:(ncol(new_products)-1)))

#Feature Engineering
new_products$TotalServiceReviews <- new_products$PositiveServiceReview + new_products$NegativeServiceReview

#The bestsellers column will, thus, be removed completely
new_products$BestSellersRank <- NULL

predictions_new_products <- predict(model, new_products)
predictions_new_products
new_products$Volume <- predictions_new_products


#Predictions by product type
#Feature Engineering
new_products <- new_products[new_products$ProductTypeLaptop == 1 | new_products$ProductTypePC == 1 | new_products$ProductTypeNetbook == 1 | new_products$ProductTypeSmartphone == 1,]

new_products_undummy$Volume <- predictions_new_products

new_products_undummy <- new_products_undummy[new_products_undummy$ProductType == 'PC' | new_products_undummy$ProductType == 'Laptop' | new_products_undummy$ProductType == 'Netbook' | new_products_undummy$ProductType == 'Smartphone',]

ggplot(new_products_undummy, aes(x = ProductType,y = Volume, fill = ProductType)) +
  geom_col()

ggplot(new_products_undummy, aes(x="", y=Volume, fill=ProductType)) +
  geom_bar(width = 1, stat='identity') +
  coord_polar("y")



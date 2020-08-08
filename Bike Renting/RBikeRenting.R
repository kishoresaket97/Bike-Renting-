rm(list=ls(all=T))
setwd("C:/Users/levi0/Downloads")
getwd()

#run this line of code to install all the packages 
#install.packages(c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
                   #"MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees'))

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
rm(x)

bike_rental = read.csv("day.csv", header = T, na.strings = c(" ", "", "NA"))

bike_rental$dteday <- format(as.Date(bike_rental$dteday,format="%Y-%m-%d"), "%d")

#Removal of variables which carry no useful data for prediction
bike_rental <- subset(bike_rental, select = -c(dteday, instant, casual, registered))

#No missing values found in the dataset
missing_val = data.frame(apply(bike_rental,2,function(x){sum(is.na(x))}))

#categorical variables
cat_var = c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit')

#numerical or continous variables
cnames = c('temp', 'atemp', 'hum', 'windspeed')

#finding outliers
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bike_rental))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Counts")+
           ggtitle(paste("Box plot of counts for",cnames[i])))
}

## Plotting boxplots
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

#Replacing outliers with NAN
for(i in cnames)
{
  val = bike_rental[,i][bike_rental[,i] %in% boxplot.stats(bike_rental[,i])$out]
  print (i)
  print(length(val))
  bike_rental[,i][bike_rental[,i] %in% val] = NA
}

missing_val = data.frame(apply(bike_rental,2,function(x){sum(is.na(x))}))

#removing the rows with outliers
bike_rental <- na.omit(bike_rental)


#correlation-analysis
#Feature selection
library(corrplot)

M <- cor(bike_rental)

corrplot::corrplot(M, method = 'number')

#removing atemp variable
bike_rental <- subset(bike_rental, select = -c(atemp))


#chi-square test of independence
cat_var = c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit')
cat = bike_rental[,cat_var]

#finding the p-values and printing them 
for (i in cat_var){
  for (j in cat_var){
    print(i)
    print(j)
    print(chisq.test(table(cat[,i], cat[,j]))$p.value)
  }
}

# finding feature importance
imp <- randomForest(cnt ~ ., data = bike_rental,
                               ntree = 200, keep.forest = FALSE, importance = TRUE)
importanceF <- data.frame(importance(imp, type = 1))

#We delete holiday and month after checking the relative feature importances
bike_rental <- subset(bike_rental, select = -c(mnth, holiday))

#VIF tests
# there seems to be no multicollinearity in the dataset as all the VIF values are below 5 
library(usdm)
vif(subset(bike_rental, select = -c(cnt)))

#Normality check
qqnorm(bike_rental$hum)
hist(bike_rental$hum)

cnames = c('temp', 'hum', 'windspeed')

#Normalisation

for (i in cnames)
{print(i)
  bike_rental[,i] = (bike_rental[,i] - min(bike_rental[,i]))/
    (max(bike_rental[,i] - min(bike_rental[,i])))
}

#Clean the environment
library(DataCombine)
rmExcept("bike_rental")


#defining MAPE for evaluation before building models
mape = function(y, yhat){
  mean(abs((y - yhat)/y))
}

#Model Development

df = bike_rental


#Divide the data into train and test
#set.seed(123)
train_index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[train_index,]
test = df[-train_index,]


#Decision Tree 
fit = rpart(cnt ~ ., data = train, method = "anova")

#Prediction
predictions_DT = predict(fit, test[,-9])

#evaluation
postResample(predictions_DT, test$cnt)
mape(test$cnt, predictions_DT)

#R-Squared : 0.78
#MAPE = 0.19

#Linear Regression
#run regression model
lm_model = lm(cnt ~., data = train)

#Summary of the model
summary(lm_model)

#Predict
predictions_LR = predict(lm_model, test[,1:8])

#evaluation
postResample(predictions_LR, test$cnt)
mape(test$cnt, predictions_LR)

#R-Squared : 0.76
#MAPE = 0.19

#Random forest model
rf_model = randomForest(cnt ~. , train, importance = TRUE, ntree = 500)

predictions_RF = predict(rf_model, test[,1:8])

postResample(predictions_RF, test$cnt)

mape(test$cnt, predictions_RF)

#R-squared : 0.78
#MAPE = 0.14


#XGB Regressor
library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(train), label = train$cnt)

bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "reg:linear")

predictions_XGB <- predict(bstDMatrix, as.matrix(test))

postResample(predictions_XGB, test$cnt)
mape(test$cnt, predictions_XGB)

#R-Squared : 0.94
#MAPE : 0.10

#So far XGB Regressor has performed the best with an MAPE of 10% and R-Squared value of 0.94
#However the MAPE value of Random Forest is also pretty good, let us perform hyperparameter 
#tuning on these 2 models and check the results 

#Hyperparameter tuning Random Forest
control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train, method = "rf",trControl = control)
pred_RFH <- predict(reg_fit, test[,-9])

postResample(pred_RFH, test$cnt)
mape(test$cnt, pred_RFH)

#After hyperparameter tuning of Random Forest 
# R-Squared : 0.89
# MAPE : 0.13

#Hyperparameter tuning XGBoost
control <- trainControl(method="repeatedcv", number=10, repeats=3)
reg_fit <- caret::train(cnt~., data = train, method = "xgbTree",trControl = control)
pred_XG_H <- predict(reg_fit, test[,-9])

postResample(pred_XG_H, test$cnt)
mape(test$cnt, pred_XG_H)


#After hyperparameter tuning of XGBoost 
# R-Squared : 0.88
# MAPE :0.14

# K-Fold Cross validation for checking the stability of the XGB model 
train_control <- trainControl(method="cv", number=10)
model <- caret::train(cnt~., data=train, trControl=train_control, method="xgbTree")
pred_XG_K <- predict(model, test[,-9])
postResample(pred_XG_K, test$cnt)
mape(test$cnt, pred_XG_K)

#Random Forest performed better after hyperparameter tuning however XGB was performing better
#without hyperparameter tuning so we finalize XGB Regression without tuning as the final model 
#with R-Squared of 0.94 and an MAPE of ~10 percent after checking the the stability of model using 
#K-fold cross validation technique 

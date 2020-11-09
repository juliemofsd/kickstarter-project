###############################
####KICKSTARTER EXPLORATION####
###############################

#IMPORT DATASETS
df<-read.csv('https://raw.githubusercontent.com/CallBark/kickstarter/main/Kickstarter_NA.csv', header = T)
Testing<-read.csv('https://raw.githubusercontent.com/Dereklip/GSBA576/main/Testing_v2.csv', header = T)
Training<-read.csv('https://raw.githubusercontent.com/Dereklip/GSBA576/main/Training_v2.csv', header = T)
Validation<-read.csv('https://raw.githubusercontent.com/Dereklip/GSBA576/main/Validation_v2.csv', header = T)

#LOAD LIBRARIES#
library(caret) #data partitioning library and other machine learning tools
library(rpart) #CART library
library(e1071) #svm library
library(randomForest) #random forest

##DESCRIPTIVE SUMMARY STATISTICS##
View(df)
dim(df)
head(df)
summary(df) 
dim(Testing)
dim(Training)
dim(Validation)


##DATA VISUALIZATIONS
hist(df$USD.Pledged, prob = TRUE) #generates a histogram for the USD.Pledged variable
curve(dnorm(x, mean = mean(df$USD.Pledged), sd = sd(df$USD.Pledged)), col = "darkblue", lwd = 2, add = TRUE) #add calibrated normal density curve to histogram
plot(density(df$USD.Pledged)) #generates a nonparametric density estimate of the distribution of the usd_pledged variable
hist(df$Goal, prob = TRUE) #generates a histogram for the Goal variable
curve(dnorm(x, mean = mean(df$Goal), sd = sd(df$Goal)), col = "darkblue", lwd = 2, add = TRUE) #add calibrated normal density curve to histogram
pairs(df[,3:8]) #generate pairs of scatter plots from the variables in columns 3-8

##BUILDING A LINEAR MODEL TO QUANTIFY THE RELATIONSHIP BETWEEN USD PLEDGED AND NUMBER OF BACKERS##
M1<-lm(USD.Pledged~Backers.Count, df)  #model: USD Pledged = B_0+B_1(backers_count)+e
summary(M1) #produces the summary output of the model
confint(M1) #returns upper and lower bounds from the 95% confidence interval for each model parameter

##VISUALIZATIONS
plot(df$USD.Pledged~df$Backers.Count) #scatter plot of USD Pledged vs. Backers.Count again
abline(M1$coefficients[1], M1$coefficients[2], col='blue', lwd=2) #add regression line to plot

##PLOTTING FITTED (PREDICTED) VALUES
plot(M1$fitted.values~df$Backers.Count)
abline(M1$coefficients[1], M1$coefficients[2], col='blue', lwd=2) #add regression line to plot

##RESIDUAL ANALYSIS##
plot(M1$residuals)
abline(0,0,col='black')
hist(M1$residuals)
summary(M1$residuals)
curve(dnorm(x, mean = mean(M1$residuals), sd = sd(M1$residuals)), col = "darkblue", lwd = 2, add = TRUE) #add calibrated normal density curve to histogram

##QUESTION: HOW MUCH CAN WE EXPECT THE USD PLEDGE TO BE CONSIDERING THE VARIABLES?##
##EXPLORING UNIVARIATE REGRESSION MODELS##
M2<-lm(USD.Pledged~Blurb.Length, df)  #builds the model: USD.Pledged = B_0+B_1(Blurb.length)+e
summary(M2)  #returns summary output from the model M2

M3<-lm(USD.Pledged~Blurb.Caps, df)  #builds the model: USD.Pledged = B_0+B_1(Blurb.caps)+e
summary(M3)  #returns summary output from the model M3

M4<-lm(USD.Pledged~Name.Length, df)  #builds the model: USD.Pledged = B_0+B_1(Name.Length)+e
summary(M4)  #returns summary output from the model M4

M5<-lm(USD.Pledged~Spotlight, df)  #builds the model: USD.Pledged = B_0+B_1(spotlight)+e
summary(M5)  #returns summary output from the model M5

M6<-lm(USD.Pledged~Staff.Pick, df)  #builds the model: USD.Pledged = B_0+B_1(staff_pick)+e
summary(M6)  #returns summary output from the model M6

M7<-lm(USD.Pledged~Goal, df)  #builds the model: USD.Pledged = B_0+B_1(goal)+e
summary(M7)  #returns summary output from the model M7

##EXPLORING MULTIVARIATE REGRESSION MODELS##
#QUESTION: WILL CONTROLLING FOR SPOTLIGHT AND STAFF PICK INFLUENCE THE AMOUNT OF USD PLEDGED?
##BUILD A MULTIVARIATE MODEL TO PREDICT USD PLEDGED CONTROLLING FOR BOTH SPOTLIGHT AND STAFF PICK##
M8<-lm(USD.Pledged~Blurb.Length+Blurb.Caps+Name.Length+Backers.Count+Spotlight+Staff.Pick+Goal, df) #model: USD.Pledged = B_0+B_1(Backers.Count)+B_2(Spotlight)+B_2(staff_pick)+e
summary(M8)
cbind( M8$coefficients, confint(M8))

##SOME RESIDUAL ANALYSIS##
plot(M8$residuals) #plots residuals
abline(0,0,col='black', lwd = 4) #adds a straight line across X-axis

hist(M8$residuals, prob = TRUE) #plots histogram of residuals
#adds calibrated normal curve to residual histogram
curve(dnorm(x, mean = mean(M8$residuals), sd = sd(M8$residuals)), col = "darkblue", lwd = 2, add=TRUE)
summary(M8$residuals) #summary descriptive stats for residuals

library(tseries) #loads "tseries" library - need to first install "tseries" package
#conducts a hypothesis test for normality called the Jarque-Bera test
jarque.bera.test(M8$residuals) #null hypothesis: data is distribution is normal

#BUILD MODEL M8 WITH ONLY THE TRAINING DATA PARTITION
M9<-lm(USD.Pledged~Blurb.Length+Blurb.Caps+Name.Length+Backers.Count+Spotlight+Staff.Pick+Goal, Training) #model: USD.Pledged = B_0+B_1(Backers.Count)+B_2(Spotlight)+B_2(staff_pick)+e
summary(M9)

#RIDGE REGRESSION / no penalty to the coefficients
library(MASS)
names(Training)[1] <- "y"
lm.ridge(y ~ ., Training)
plot(lm.ridge(y ~ ., Training,
              lambda = seq(0,0.1,0.001)))
select(lm.ridge(y ~ ., Training,
                lambda = seq(0,0.1,0.0001)))

##CALCULATE ROOT MEAN SQUARE PREDICTION ERROR ON TEST DATA: THE IN-SAMPLE ERROR MEASURE
RMSE_IN<-sqrt(sum((M9$fitted.values-Testing$USD.Pledged)^2)/length(Testing$USD.Pledged))
RMSE_IN #report root mean squared error (E_out) using the out-of-sample testing data

#EVALUATE M9 ON THE TEST PARTITION TO COMPUTE THE OUT-OF-SAMPLE PREDICTIONS
predictions<-predict(M9, Testing)
View(predictions) # view predictions for December

##CALCULATE ROOT MEAN SQUARE PREDICTION ERROR ON TEST DATA: THE OUT-OF-SAMPLE ERROR MEASURE
RMSE_OUT<-sqrt(sum((predictions-Testing$USD.Pledged)^2)/length(Testing$USD.Pledged))
RMSE_OUT #report root mean squared error (E_out) using the out-of-sample testing data

#####################
#LOGISTIC REGRESSION#
#####################

Training$State<-factor(Training$State) #convert State to factor (categorical) variable
Training$Category<-factor(Training$Category) #convert Category to factor (categorical) variable
Training$Spotlight<-factor(Training$Spotlight) #convert Spotlight to factor (categorical) variable
Training$Staff.Pick<-factor(Training$Staff.Pick) #convert Spotlight to factor (categorical) variable

Testing$State<-factor(Testing$State) #convert State to factor (categorical) variable
Testing$Category<-factor(Testing$Category) #convert Category to factor (categorical) variable
Testing$Spotlight<-factor(Testing$Spotlight) #convert Spotlight to factor (categorical) variable
Testing$Staff.Pick<-factor(Testing$Staff.Pick) #convert Spotlight to factor (categorical) variable


##(need State to be factor)
M_LOG<-glm(State ~ Backers.Count, data = Training, family = "binomial")
summary(M_LOG)
exp(cbind(M_LOG$coefficients, confint(M_LOG)))
confusionMatrix(table(predict(M_LOG, Testing, type="response") >= 0.5,
                      Testing$State == 1), positive='TRUE')

#####################
########CART#########
#####################

#rpart package implementation
train_control <- trainControl(method="cv", number=10, savePredictions = TRUE)
M_CART <- train(State ~ Backers.Count, data = Training, trControl=train_control, tuneLength=10, method = "rpart") #increasing tunelength increases regularization penalty
##the "cv", number = 10 refers to 10-fold cross validation on the training data
plot(M_CART) #produces plot of cross-validation results
M_CART$bestTune #returns optimal complexity parameter
confusionMatrix(predict(M_CART, Testing), Testing$State, positive='1')


#####################
####RANDOM FOREST####
#####################

#caret package implementation with 10-fold cross validation

train_control <- trainControl(method="cv", number=10, savePredictions = TRUE)
RF1 <- train(State ~ Backers.Count, method="rf", trControl=train_control, preProcess=c("center", "scale"), tuneLength=2, data=Training)
print(RF1)
confusionMatrix(predict(RF1, Testing), Testing$State, positive='1')

#random forest package implementation
RF2 <- randomForest(State ~ Backers.Count, Training)
print(RF2)
confusionMatrix(predict(RF2, Testing), Testing$State, positive='1')

########################
#SUPPORT VECTOR MACHINE#
########################

Training$State<-factor(Training$State)
Testing$Backers.Count<-factor(Testing$Backers.Count)
Training$State<-as.numeric(Training$State)
Testing$Backers.Count<-as.numeric(Testing$Backers.Count)

#e1071 package implementation
set.seed(123)
SVM1<-svm(State~Backers.Count, data = Training)
confusionMatrix(predict(SVM1, Testing), Testing$State, positive='1')

#tuning the SVM (validation)
set.seed(123)
svm_tune <- tune(svm, train.x=Training[,-1], train.y=Training[,1], 
                 ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune) #recover optimal gamma and cost parameters from validation

#re-estimate the model with the optimally tuned parameters
SVM_RETUNE<-svm(State~Backers.Count, data = Training, cost=1, gamma=.5)
confusionMatrix(predict(SVM_RETUNE, Testing), Testing$State, positive='1')



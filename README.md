###############################
####KICKSTARTER EXPLORATION####
###############################

#IMPORT DATASET
df<-read.csv('https://raw.githubusercontent.com/juliemofsd/kickstarter-project/main/Kickstarter_JMorgan.csv')

##DESCRIPTIVE SUMMARY STATISTICS##
View(df)
dim(df)
head(df)
summary(df)
cov(df[,2:10]) #generates the variance-covariance matrix from column variables 2-10
cor(df[,2:10]) #generates the correlation matrix for column variables 2-10

#CLEANING THE DATA
kickstarter_1<-df  ##create new dataset from starting point
kickstarter_1$State[kickstarter_1$State>1]<-NA ##assigns NA values for invalid responses
dim(kickstarter_1)

kickstarter_2<-subset(kickstarter_1, State<1) ##deletes observations with invalid State response
dim(kickstarter_2)

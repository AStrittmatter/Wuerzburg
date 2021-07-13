########################  Load Packages and Data  ########################

# Load packages
library(rpart)
library(rpart.plot)
library(grf)
library(glmnet)
library(nnet)

# Load data
cars <- read.csv("used_car_database.csv", sep = ",")
new_cars <- read.csv("new_used_cars.csv", sep = ",")

print('Packages and data successfully loaded.')

#############################################################################

########################  Describe Old Data  ########################

# Print first few rows of old data
head(cars)

# Number of observations
print(paste0('Old data: ',nrow(cars),' observations'))

######################################################################

########################  Describe New Data  ########################

# Print first few rows of new data
head(new_cars)

# Number of observations
print(paste0('New data: ',nrow(new_cars),' observations'))

######################################################################

########################  Data Preparation  ########################

# Generate outcome and control variables
y <- as.matrix(cars[,2])
x <- as.matrix(cars[,-c(1:2)])
new_x <- as.matrix(new_cars[,-1])

print('Data is prepared.')

#############################################################################

########################  Training and Test Samples  ########################

set.seed(???)

# Generate variable with the rows in training data
???

print('Training and test samples created.')

#############################################################################

########################  LASSO, Ridge, Elastic Net  ##############################


                                   
################################################################

######################  Regression Tree  #######################



################################################################

########################  Random Forest  #######################


################################################################

########################  Out-of-Sample Prediction  #######################

# Fitted values
new_prediction <- ???

print('Out-of-sample sales are predicted.')

###########################################################################

########################  Store Results  #######################

id_new <- as.matrix(new_cars$id)

# Replace ??? with your last name
write.csv(cbind(id_new,new_prediction),"???.csv")

print('File is stored.')
print('Upload your results, code and answer to WueCampus')

################################################################



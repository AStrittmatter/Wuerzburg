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

set.seed(1001)

# Generate variable with the rows in training data
size <- floor(0.5 * nrow(cars))
training_set <- sample(seq_len(nrow(cars)), size = size)

print('Training and test samples created.')

#############################################################################

########################  LASSO, Ridge, Elastic Net  ##############################

set.seed(27112019)
penalized.cv <- cv.glmnet(x[training_set,],y[training_set], 
                      type.measure = "mse", family = "gaussian", nfolds = 10, alpha = 0.5)


# Fitted values
pred_penalized <- predict(penalized.cv, newx = x, s = penalized.cv$lambda.min, alpha = 0.5)

# Calculate the MSE
MSE_penalized <- mean((y[-training_set] - pred_penalized[-training_set])^2)
R2_penalized <- round(1- MSE_penalized/var(y[-training_set]), digits = 3)

print(paste0("R-squared Penalized Regression: ", R2_penalized))
                                   
################################################################

######################  Regression Tree  #######################

set.seed(27112019)
# Prepare data for tree estimator
outcome <- y[training_set]
tree_data <- data.frame(outcome, x[training_set,])

deep_tree <- rpart(formula = outcome ~ ., data = tree_data, method = "anova",
                xval = 10, y = TRUE, control = rpart.control(cp = 0.00002, minbucket=10))

# Optimal tree size
op.index <- which.min(deep_tree$cptable[, "xerror"])

## Select the Tree that Minimises CV-MSE
cp.vals <- deep_tree$cptable[op.index, "CP"]

# Prune the deep tree
pruned_tree <- prune(deep_tree, cp = cp.vals)

## Plot tree structure
#rpart.plot(pruned_tree,digits=3)

# Fitted values
predtree <- predict(pruned_tree, newdata= as.data.frame(x))

# Calculate the MSE
MSEtree <- mean((y[-training_set] - predtree[-training_set])^2)
R2tree <- round(1- MSEtree/var(y[-training_set]), digits = 3)

print(paste0("R-squared Tree: ", R2tree))

################################################################

########################  Random Forest  #######################

set.seed(27112019)

rep <- 1000 # number of trees
cov <- 2/3 # share of covariates
frac <- 1/2 # fraction of subsample
min_obs <- 10 # max. size of terminal leaves in trees

# Build Forest
forest <- regression_forest(x[training_set,],y[training_set,], mtry = floor(cov*ncol(x)), sample.fraction = frac,
            num.trees = rep,min.node.size = min_obs, honesty=FALSE)

# Fitted values
predforest <- predict(forest, newdata=x)$predictions

# Calculate MSE
MSEforest <- mean((y[-training_set] - predforest[-training_set])^2)
R2forest <- round(1- MSEforest/var(y[-training_set]), digits = 3)

print(paste0("R-squared Forest: ", R2forest))

################################################################

########################  Out-of-Sample Prediction  #######################

# Fitted values
# We select Random Forest because highest R-squared
new_prediction <- predict(forest, newdata=new_x)$predictions

print('Out-of-sample sales are predicted.')

###########################################################################

########################  Store Results  #######################

id_new <- as.matrix(new_cars$id)

# Replace ??? with your last name
write.csv(cbind(id_new,new_prediction),"strittmatter.csv")

print('File is stored.')
print('Upload your results, code and answer to WueCampus')

################################################################



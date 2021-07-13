########################  Load Packages and Data  ########################

# Load packages
library(rpart)
library(rpart.plot)
library(grf)
library(glmnet)

# Load data
juice <- read.csv("juice.csv", sep = ",")
new_grocery <- read.csv("new_grocery.csv", sep = ",")

print('Packages and data successfully loaded.')

#############################################################################

########################  Describe Old Data  ########################

# Print first few rows of old data
head(juice)

# Number of observations
print(paste0('Old data: ',nrow(juice),' observations'))

######################################################################

########################  Describe Old Data  ########################

# Print first few rows of new data
head(new_grocery)

# Number of observations
print(paste0('New data: ',nrow(new_grocery),' observations'))

######################################################################

########################  Data Preparation  ########################

# Generate dummy for missing prices
missing <- (is.na(juice$price) == TRUE)
new_missing <- (is.na(new_grocery$price) == TRUE)

# Replace missing prices with zero
juice$price[is.na(juice$price)] <-0
new_grocery$price[is.na(new_grocery$price)] <-0

# Generate Dummies for Brands
brand_1 <- (juice$brand == "minute.maid")
brand_2 <- (juice$brand == "dominicks")
brand_3 <- (juice$brand == "tropicana")

new_brand_1 <- (new_grocery$brand == "minute.maid")
new_brand_2 <- (new_grocery$brand == "dominicks")
new_brand_3 <- (new_grocery$brand == "tropicana")

# Generate outcome and control variables
y <- as.matrix(juice$sales)
colnames(y) <- c("sales")

x <- as.matrix(cbind(juice$price, missing, brand_1, brand_2, brand_3, juice$feat))
colnames(x) <- c("price", "missing", "minute.maid", "dominicks", "tropicana", "featured")

new_x <- as.matrix(cbind(new_grocery$price, new_missing, new_brand_1, new_brand_2, new_brand_3, new_grocery$feat))
colnames(new_x) <- c("price", "missing", "minute.maid", "dominicks", "tropicana", "featured")

# Descriptive statistics
summary(cbind(y,x))

print('Data is prepared.')

#############################################################################

########################  Training and Test Samples  ########################

set.seed(1001)

# Generate variable with the rows in training data
size <- floor(0.5 * nrow(juice))
training_set <- sample(seq_len(nrow(juice)), size = size)

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
                xval = 10, y = TRUE, control = rpart.control(cp = 0.00002, minbucket=100))

# Optimal tree size
op.index <- which.min(deep_tree$cptable[, "xerror"])

## Select the Tree that Minimises CV-MSE
cp.vals <- deep_tree$cptable[op.index, "CP"]

# Prune the deep tree
pruned_tree <- prune(deep_tree, cp = cp.vals)

## Plot tree structure
rpart.plot(pruned_tree,digits=3)

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
forest <- regression_forest(x[training_set,], y[training_set,], mtry = floor(cov*ncol(x)), sample.fraction = frac,
            num.trees = rep, min.node.size = min_obs, honesty=FALSE)

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
new_prediction <- predict(forest, newdata = new_x)$predictions

print('Out-of-sample sales are predicted.')

###########################################################################

########################  Store Results  #######################

id_new <- as.matrix(new_grocery$id)

# Replace ??? with your last name
write.csv(cbind(id_new,new_prediction),"strittmatter.csv")

print('File is stored.')

################################################################



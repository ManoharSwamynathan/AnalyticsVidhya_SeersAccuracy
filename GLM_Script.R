# -----------------------------------
# Load libraries
# -----------------------------------
library(lubridate)
library(eeptools)
library(dummies)
library(plyr)
library(caret)
library(Epi)
library(pROC)

# -----------------------------------
# set working directory & load data
# -----------------------------------
setwd('C:\\Users\\Manohar Swamynathan\\Documents\\AnalyitcsVidya\\Script')
data <- read.csv('../data/Train_seers_accuracy.csv', sep=',', header=T)

head(data)

# -----------------------------------
# Data Preparation
# -----------------------------------
data$Transaction_Date <- as.Date(as.POSIXlt(data$Transaction_Date, format="%m/%d/%Y"))
data$DOB <- as.Date(as.POSIXlt(data$DOB, format="%m/%d/%Y"))
data$month <- month(as.Date(as.POSIXlt(data$Transaction_Date, format="%m/%d/%Y")))

# Clean DOB that has future dates i.e., 2026 & 2028 as year
data$DOB <- ifelse(data$DOB > Sys.Date(), as.Date(as.POSIXlt(Sys.Date(), format="%m/%d/%Y")), as.Date(as.POSIXlt(data$DOB, format="%m/%d/%Y")))
data$DOB <- as.Date(data$DOB, origin = "1970-01-01")

# Calculate age of customers
data$age <- floor(age_calc(as.Date(data$DOB), enddate = as.Date(Sys.Date()), units = "years",precise=T))

# Add visit count and revisit flag for records
data <- arrange(data, Client_ID, Transaction_Date) 
data$Revisit <- duplicated(data$Client_ID)
data$Revisit <- ifelse(data$Revisit == FALSE, 0, 1)
data$visit_count <- sequence(rle(data$Client_ID)$lengths)

# Convert categorical values to numeric values
data$Purchased_in_Sale <- ifelse(data$Purchased_in_Sale == 'N', 0, 1)
data$Referred_Friend <- ifelse(data$Referred_Friend == 'NO', 0, 1)

# Create dummy variables for categorical variables
data <- cbind(data, dummy(data$Gender))
names(data) <- gsub("data","gender_",names(data))
data <- cbind(data, dummy(data$Sales_Executive_Category))
names(data) <- gsub("data","SalesExeCat",names(data))
data <- cbind(data, dummy(data$Lead_Source_Category))
names(data) <- gsub("data","SourceCat_",names(data))
data <- cbind(data, dummy(data$Payment_Mode))
names(data) <- gsub("data","PayMode_",names(data))
data <- cbind(data, dummy(data$Product_Category))
names(data) <- gsub("data","ProdCat_",names(data))
names(data) <- gsub(" ","_",names(data)) # Replace space with underscore
data <- cbind(data, dummy(data$Number_of_EMI))
names(data) <- gsub("data","No_EMI_",names(data))
data <- cbind(data, dummy(data$Var1))
names(data) <- gsub("data","Var1_",names(data))
data <- cbind(data, dummy(data$Var2))
names(data) <- gsub("data","Var2_",names(data))
data <- cbind(data, dummy(data$Var3))
names(data) <- gsub("data","Var3_",names(data))
data <- cbind(data, dummy(data$month))
names(data) <- gsub("data","month_",names(data))

################## Create target variable i.e., revisit within 12 months flag ###################
diffs <- function(r, order.by, col) {
  order.by <- order.by[r]
  col <- col[r]
  o <- order(order.by)
  replace(r, o, c(NA, diff(col[o])))
}

# fun specialized to arguments after first, i.e. subsequent arguments curried
curry <- function (fun, ...) function(r) fun(r, ...)

ix <- 1:nrow(data)
data <- transform(data, 
                  days_since = ave(ix, Client_ID, FUN = curry(diffs, Transaction_Date, Transaction_Date)))

# if revisit is true and days_since is less than 365, flag it as cust returned within 12 months
data$revisit_in_12_months <- ifelse(data$Revisit == 1 & data$days_since <= 365, 1, 0)


# Delete unwanted columns
data$Sales_Executive_Category <- NULL
data$Lead_Source_Category <- NULL
data$Payment_Mode <- NULL
data$Product_Category <- NULL
data$Gender <- NULL
data$gender__ <- NULL
data$PayMode_ <- NULL
data$Number_of_EMI <- NULL
data$Var1 <- NULL
data$Var2 <- NULL
data$Var3 <- NULL
data$Transaction_Date <- NULL
data$month <- NULL
data$DOB <- NULL

# Split data to train and test
set.seed(1234)
splitIndex <- createDataPartition(data$revisit_in_12_months, p = .80, list = FALSE, times = 1)
train <- data[ splitIndex,]
test <- data[-splitIndex,]

# check target distribution between train and test
prop.table(table(train$revisit_in_12_months))
prop.table(table(test$revisit_in_12_months))

target <- 'revisit_in_12_months'
feature.names <- c('Purchased_in_Sale', 'Referred_Friend', 'Transaction_Amount', 'age', 'Revisit', 'gender_C', 'gender_F', 'gender_M', 'SalesExeCatA', 'SalesExeCatB', 'SalesExeCatC', 'SalesExeCatD', 'SalesExeCatE', 'SourceCat_Advertisment', 'SourceCat_Other', 'SourceCat_Reference', 'SourceCat_Walkin', 'PayMode_Cash', 'PayMode_Cheque', 'PayMode_Credit.Debit_Card', 'PayMode_Other', 'ProdCat_Cat_A', 'ProdCat_Cat_B', 'ProdCat_Cat_C', 'ProdCat_Cat_D', 'ProdCat_Cat_E', 'ProdCat_Cat_F', 'ProdCat_Cat_G', 'ProdCat_Cat_H', 'No_EMI_1', 'No_EMI_2', 'No_EMI_3', 'No_EMI_6', 'No_EMI_8', 'Var1_1', 'Var1_2', 'Var1_3', 'Var2_1', 'Var2_2', 'Var3_1', 'Var3_2', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12')

returnsig <- function(target, independents, data){
  fit <- lm(paste(target, '~', paste(independents, collapse = " + ")), data = data)
  sig <- summary(fit)$coeff[-1,4] < 0.05
  sig <- names(sig[sig == TRUE])
  return(sig)
}

# Identify significant variables
sig <- returnsig(target ,feature.names, train)

fm <- paste(target, '~', paste(sig, collapse = " + "))
fm <- paste(target, '~', paste(feature.names, collapse = " + "))

# create model using generalized linear model
# ----------------------------------------------------
x.glm <- glm(fm, train, family = binomial("logit"))
summary(x.glm)

train$glm.prob <- predict(x.glm, train, type="response")
test$glm.prob <- predict(x.glm, test, type="response")
data$glm.prob <- predict(x.glm, data, type="response")

confusionMatrix(table(train$glm, train$revisit_in_12_months))
confusionMatrix(table(test$glm, test$revisit_in_12_months))
confusionMatrix(table(data$Cross_Sell, data$revisit_in_12_months))

auc(data$Cross_Sell, data$revisit_in_12_months)

ROC(form = fm, data = train)

submission <- subset(data, select = c('Client_ID', 'glm.prob'))
submission <- submission[!duplicated(submission[,c('Client_ID')]),]
colnames(submission)[colnames(submission)=="glm.prob"] <- "Cross_Sell"
write.csv(submission, file = 'Sample_Submission.csv', row.names = F)

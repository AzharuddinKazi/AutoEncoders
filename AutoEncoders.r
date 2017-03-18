rm(list=ls(all=T))

### set the working directory
setwd("C:/Users/Dell/Downloads/20170318_Batch23_7219c_GNQ_Datasets/20170318_Batch23_7219c_GNQ_Datasets")

### use read.table to read the text file in r
bank_note <- read.table("data_banknote_authentication.txt", header = FALSE, sep = ",")

### change the column name of target variable
colnames(bank_note)[5] <- "Class"

### check the structure of dataset
str(bank_note)

### as we have to predict the class to be 0 or 1, convert the class variable to factor
bank_note$Class <- as.factor(bank_note$Class)

### load H2o
require(h2o)
localh2o <- h2o.init()

### convert into h2o objects to supply to autoencoders
bank_note.hex <- as.h2o(client = localh2o, object = bank_note, key = "train.hex")


### giving the data to autoencoder
aec <- h2o.deeplearning(x = setdiff(colnames(bank_note.hex),"Class"),
                        y = "Class", data = bank_note.hex,
                        autoencoder = T,
                        activation = "RectifierWithDropout",
                        classification = T,
                        hidden = 10,
                        epochs = 100,
                        l1 = 0.01)

### extracting features from autoencoders 
features_bnk_nt <- as.data.frame.H2OParsedData(h2o.deepfeatures(bank_note.hex[,-5], model = aec))
bnk_nt_fnl <- data.frame(cbind(bank_note[,-5],features_bnk_nt,Class = bank_note[,5]))

### divide into train and test by taking stratified sampling
require(caret)
datapart <- createDataPartition(bnk_nt_fnl$Class, p=0.7, list = FALSE)
train <- bnk_nt_fnl[datapart,]
test <- bnk_nt_fnl[-datapart,]

### convert into h2o objects
train.hex <- as.h2o(client = localh2o, object = train, key = "train.hex")
test.hex <- as.h2o(client = localh2o, object = test, key = "test.hex")

### train the deeplearning model
model <- h2o.deeplearning(x = setdiff(colnames(bank_note.hex),"Class"),
                          y = "Class",
                          data = train.hex,
                          hidden = c(5,5,5),
                          activation = "RectifierWithDropout",
                          input_dropout_ratio =0.1,
                          epochs = 100,
                          seed = 123)

### predict on train and test
prediction_train <- h2o.predict(model, newdata = train.hex)
# convert h2o object back to data frame
pred_train <- as.data.frame(prediction_train) 

prediction_test = h2o.predict(model, newdata = test.hex)
# convert h2o object back to data frame
pred_test = as.data.frame(prediction_test)

# confusion matrix and evaluatioon metric for train
cnf_mat_tr <- table(train$Class,pred_train$predict) 
cnf_mat_tr

accuracy_train = sum(diag(cnf_mat_tr))/sum(cnf_mat_tr)
precision_train = cnf_mat_tr[2,2]/sum(cnf_mat_tr[,2])
recall_Train = cnf_mat_tr[2,2]/sum(cnf_mat_tr[2,])

# Confusion Matrix and evaluation metric for test
cnf_mtr_tst=table(test$Class, pred_test$predict)
cnf_mtr_tst

accuracy_test = sum(diag(cnf_mtr_tst))/sum(cnf_mtr_tst)
precision_test = cnf_mtr_tst[2,2]/sum(cnf_mtr_tst[,2])
recall_Test = cnf_mtr_tst[2,2]/sum(cnf_mtr_tst[2,])

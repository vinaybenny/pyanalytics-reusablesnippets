library(caret);
library(reshape2);
library(mlbench);
library(e1071);
library(pROC);
library(randomForest);

set.seed(12345);

# Check corelations and plot correlation matrix
correlationMatrix <- cor(X_train);
upper_tri <- get_upper_tri(correlationMatrix);
melted_cormat <- melt(correlationMatrix);
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value)) + geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed();

# Find candidate attributes to remove
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5);
print(colnames(X_train)[highlyCorrelated]);

# Use Leaning Vector Quantisation method to rank features
# Prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3);
# Train the model
temp <- as.data.frame(X_train);
temp$y <- as.factor(y_train);
model <- train(y~., data=temp, method="lvq", preProcess="scale", trControl=control);
# Estimate variable importance
importance <- varImp(model, scale=FALSE);
# Summarize importance
print(importance);
# Plot importance
plot(importance);

# Define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# Run the Recursive Feature Selection(RFE) algorithm
results <- rfe(as.data.frame(X_train), as.factor(y_train), sizes=c(1:8), rfeControl=control);
# Summarize the results
print(results);
# List the chosen features
predictors(results)
# Plot the results
plot(results, type=c("g", "o"));

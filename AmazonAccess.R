library(tidyverse)


library(vroom)
rm(list = objects()) # removes all objects in the environment

traindat <- vroom("train.csv")
testdat <- vroom("test.csv")

traindat$ACTION <- as.factor(traindat$ACTION)

# EDA ---------------------------------------------------------------------
library(ggmosaic)
library(ggplot2)

glimpse(traindat)
DataExplorer::plot_correlation(traindat)

ggplot(data=traindat, mapping = aes(x=MGR_ID, y=ACTION)) + 
  geom_bar(stat="identity", color="steelblue")

ggplot(data=traindat) +
  geom_box(mapping = aes(x=RESOURCE, y=ACTION), color="gold")

ggplot(data=traindat) + geom_mosaic(aes(x=product(<catvariable>), fill=ACTION))


# Logistic Regression -----------------------------------------------------
library(tidymodels)

# Feature Engineering
my_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
  step_other(all_nominal_predictors(), threshold = .01) %>% # lumps factors together with too few datapoints
  step_dummy(all_nominal_predictors()) 

prepped <- prep(my_recipe)
baked <- bake(prepped, traindat) # the data you want to clean

traindat$ACTION <- as.factor(traindat$ACTION)

logRegModel <- logistic_reg() %>%
  set_engine("glm") 

logRegWF <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=traindat)

logRpreds <- predict(logRegWF, new_data = testdat, 
                     type = "prob") # type: classification/probability

kaggle_submission <- logRpreds %>%
  bind_cols(., testdat) %>% # bind predictions with test data
  select(id, .pred_1) %>% # keep only datetime and prediction variables
  rename(ACTION=.pred_1) 
"this previously results in two columns: probability of response = 0, prob. of response = 1"

vroom_write(kaggle_submission,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Penalized Logistic Regression -------------------------------------------
library(tidymodels)
library(embed) # for target encoding

# Feature Engineering
tencode_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # *target encoding*
  step_normalize(all_nominal_predictors()) # normalize for penalized regression
  # no need for step other since target encoding doesn't overfit

pen_log_mod <- logistic_reg(mixture=tune(),
                            penalty = tune()) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(tencode_recipe) %>%
  add_model(pen_log_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 7) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 5, repeats =1) # K-folds

cv_results <- pen_log_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy))

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- pen_log_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)
  
pen_log_preds_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(pen_log_preds_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Parallel Computing ------------------------------------------------------
library(doParallel)

num_cores <- detectCores() #4
clstr <- makePSOCKcluster(num_cores)
registerDoParallel(clstr)

"code here"

stopCluster(clstr)


# KNN Models --------------------------------------------------------------
library(tidymodels)

dummy_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
  step_other(all_nominal_predictors(), threshold = .01) %>% # lumps factors together with too few datapoints
  step_dummy(all_nominal_predictors()) %>%
  #step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # *target encoding*
  step_normalize(all_nominal_predictors()) # normalize for KNN
  # no need for step other since target encoding doesn't overfit

knn_model <- nearest_neighbor(neighbors=round(sqrt(length(traindat)))) %>% # set or tune K
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(dummy_recipe) %>%
  add_model(knn_model) %>%
  fit(data=traindat)

# final_knn <- knn_wf %>%
#   finalize_workflow(bestTune) %>% 
#   fit(data=traindat)

knn_submit <- knn_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(knn_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Regression Trees Classification -----------------------------------------
library(tidymodels)
library(embed)

# Feature Engineering
tree_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # *target encoding*
   step_normalize(all_nominal_predictors()) #normalize for random forests

# ???
# tree_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
#   step_mutate_at(all_numeric_predictors(), fn = factor)

tree_mod <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=5) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_mod)

tuning_grid <- grid_regular(finalize(mtry(), traindat),
                            min_n(),
                            levels = 2) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 2, repeats =1) # K-folds

cv_tree_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy))

bestTree <- cv_tree_results %>%
  select_best(metric="roc_auc")

final_wf <- tree_wf %>%
  finalize_workflow(bestTree) %>% 
  fit(data=traindat)

tree_submit <- final_wf %>% 
  predict(new_data= testdat, type="class") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

vroom_write(tree_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Naive Bayes -------------------------------------------------------------
library(tidymodels)
library(embed)
library(discrim) # for naivebayes model

# Feature Engineering
nb_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # *target encoding*
  step_normalize(all_nominal_predictors()) #normalize for random forests

## Naive Bayes model
nb_model <- naive_Bayes(Laplace = tune(), # parameter that weights prior vs. posterior probs.
                        smoothness=tune()) %>% #density smoothness, ie. bin width for datapoints
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

# tune parameters
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 2) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 2, repeats =1) # K-folds

cv_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

nb_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

vroom_write(nb_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Principal Component Analysis -----------------------------------------------------
# rerun all previous models with principal component analysis to check for improvements

pc_recipe <- recipe(ACTION~., traindat) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_predictors()) %>% # normalize everything for eigenvectors
  step_pca(all_predictors(), threshold = 0.9) # set the threshold (reduction of features)

# view the dataset to see the reduction of dimensions
prepped <- prep(pc_recipe)
baked <- bake(prepped, traindat)
baked

## Naive Bayes model
nb_model <- naive_Bayes(Laplace = tune(), # parameter that weights prior vs. posterior probs.
                        smoothness=tune()) %>% #density smoothness, ie. bin width for datapoints
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(pc_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 7, repeats =1) # K-folds

cv_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

nb_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

vroom_write(nb_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# KNN 
knn_model <- nearest_neighbor(neighbors=round(sqrt(length(traindat)))) %>% # set or tune K
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(pc_recipe) %>%
  add_model(knn_model) %>%
  fit(data=traindat)

knn_submit <- knn_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(knn_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")

# Pen Log Res
pen_log_mod <- logistic_reg(mixture=tune(),
                            penalty = tune()) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(pc_recipe) %>%
  add_model(pen_log_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 7) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 5, repeats =1) # K-folds

cv_results <- pen_log_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) # metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- pen_log_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

pen_log_preds_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(pen_log_preds_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# Log Reg
logRegModel <- logistic_reg() %>%
  set_engine("glm") 

logRegWF <- workflow() %>%
  add_recipe(pc_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=traindat)

logRpreds <- predict(logRegWF, new_data = testdat, 
                     type = "prob") # type: classification/probability

kaggle_submission <- logRpreds %>%
  bind_cols(., testdat) %>% # bind predictions with test data
  select(id, .pred_1) %>% # keep only datetime and prediction variables
  rename(ACTION=.pred_1) 
"this previously results in two columns: probability of response = 0, prob. of response = 1"

vroom_write(kaggle_submission,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked







# SVM ---------------------------------------------------------------------

# Support Vector Models
library(tidymodels)
library(kernlab)

pc_recipe <- recipe(ACTION~., traindat) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_predictors()) %>% # normalize everything for eigenvectors
  step_pca(all_predictors(), threshold = 0.95) # set the threshold (reduction of features)

svmLinear <- svm_linear(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")
  
svmPoly <- svm_poly(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(pc_recipe) %>%
  add_model(svmLinear)

# tune parameters
tuning_grid <- grid_regular(cost(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 7, repeats =1) # K-folds

cv_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

svm_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

vroom_write(svm_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# SMOTE -------------------------------------------------------------------
# rerun everything with balanced data

library(tidymodels)
library(embed)
library(themis) # smote

smote_recipe <- recipe(ACTION~., data = traindat) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # feature encoding
  step_normalize(all_predictors()) %>% # normalize everything for eigenvectors
  step_pca(all_predictors(), threshold = 0.95) %>% # optional: use pca (reduction of features)
  step_smote(all_outcomes(), neighbors=4) # K neighbors
  # OR step_upsample(all_outcomes()) OR step_downsample(all_outcomes())
  
# view results
preppers <- prep(smote_recipe)
bake(preppers, traindat)

###
# Log Reg
logRegModel <- logistic_reg() %>%
  set_engine("glm") 

logRegWF <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=traindat)

logRpreds <- predict(logRegWF, new_data = testdat, 
                     type = "prob") # type: classification/probability

kaggle_submission <- logRpreds %>%
  bind_cols(., testdat) %>% # bind predictions with test data
  select(id, .pred_1) %>% # keep only datetime and prediction variables
  rename(ACTION=.pred_1) 
"this previously results in two columns: probability of response = 0, prob. of response = 1"

vroom_write(kaggle_submission,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# Pen Log Res
pen_log_mod <- logistic_reg(mixture=tune(),
                            penalty = tune()) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(pen_log_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 7) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 20, repeats =1) # K-folds

cv_results <- pen_log_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) # metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- pen_log_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

pen_log_preds_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(pen_log_preds_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# random forest
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees=1000) %>% # number trees
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(finalize(mtry(), traindat),
                            min_n(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 10, repeats =1) # K-folds

cv_rf_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTree <- cv_rf_results %>%
  select_best(metric="roc_auc")

final_wf <- rf_wf %>%
  finalize_workflow(bestTree) %>% 
  fit(data=traindat)

rf_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

stopCluster(clstr)

vroom_write(rf_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# KNN 
knn_model <- nearest_neighbor(neighbors=round(sqrt(length(traindat)))) %>% # set or tune K
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(knn_model) %>%
  fit(data=traindat)

knn_submit <- knn_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(knn_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

## Naive Bayes model
nb_model <- naive_Bayes(Laplace = tune(), # parameter that weights prior vs. posterior probs.
                        smoothness=tune()) %>% #density smoothness, ie. bin width for datapoints
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 10, repeats =1) # K-folds

cv_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=traindat)

nb_submit <- final_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,ACTION)

vroom_write(nb_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",") # worked

# BART
library(dbarts)

smote_recipe

bart_model <- bart(
  mode = "regression",
  engine = "dbarts",
  trees = 1000,
  prior_terminal_node_coef = .95,
  prior_terminal_node_expo = 2,
  prior_outcome_range = 2
) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>% 
  add_model(bart_model) %>%
  add_recipe(smote_recipe) %>%
  fit(data=traindat)

bart_submit <- bart_wf %>% 
  predict(new_data= testdat, type="prob") %>%
  bind_cols(testdat) %>%
  rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
  select(id,ACTION)

vroom_write(bart_submit,"./AmazonAccess/amazonSubmission.csv", delim = ",")


# Final model (w/o pca) ---------------------------------------------------



library(tidymodels)
library(embed)
library(ranger)
library(vroom)
# using SMOTE balancing
library(themis) # smote

traindat <- vroom("train.csv")
testdat <- vroom("test.csv")

# Feature Engineering ----------------------------------------------------------------
traindat$ACTION <- as.factor(traindat$ACTION)

# Logistic Regression -----------------------------------------------------
#library(tidymodels)

# Feature Engineering
# my_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
#   step_other(all_nominal_predictors(), threshold = .01) %>% # lumps factors together with too few datapoints
#   step_dummy(all_nominal_predictors()) 
# 
# prepped <- prep(my_recipe)
# baked <- bake(prepped, traindat) # the data you want to clean
# 
# traindat$ACTION <- as.factor(traindat$ACTION)
# 
# logRegModel <- logistic_reg() %>%
#   set_engine("glm") 
# 
# logRegWF <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logRegModel) %>%
#   fit(data=traindat)
# 
# logRpreds <- predict(logRegWF, new_data = testdat, 
#                      type = "prob") # type: classification/probability




# Penalized Logistic Regression -------------------------------------------
# library(embed) # for target encoding
# 
# # Feature Engineering
# tencode_recipe <- recipe(ACTION~., traindat) %>% # use traindat dataset as a template
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>% # convert numerics to factor
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # *target encoding*
#   step_normalize(all_nominal_predictors()) # normalize for penalized regression
# # no need for step other since target encoding doesn't overfit
# 
# pen_log_mod <- logistic_reg(mixture=tune(),
#                             penalty = tune()) %>%
#   set_engine("glmnet")
# 
# pen_log_wf <- workflow() %>%
#   add_recipe(tencode_recipe) %>%
#   add_model(pen_log_mod)
# 
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 7) # grid of L^2 tuning possibilities
# 
# folds <- vfold_cv(traindat, v = 5, repeats =1) # K-folds
# 
# cv_results <- pen_log_wf %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy))
# 
# bestTune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# final_wf <- pen_log_wf %>%
#   finalize_workflow(bestTune) %>% 
#   fit(data=traindat)
# 
# SFTP_export <- final_wf %>% 
#   predict(new_data= testdat, type="prob") %>%
#   bind_cols(testdat) %>%
#   rename(ACTION=.pred_1) %>% # pred_1 is prediction on response = 1, pred_0 for respones=0
#   select(id,ACTION)
# 
# 
# vroom_write(SFTP_export,"./AmazonAccess/amazonSubmission.csv", delim = ",")
# save(file="./amazonSFTP.RData", list=c("final_wf","SFTP_export"))


# Regression Trees Classification -----------------------------------------


rf_recipe <- recipe(ACTION~., data = traindat) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% # feature encoding
  step_normalize(all_nominal_predictors()) #%>% # or normalize all_predictors for smote
  #step_pca(all_predictors(), threshold = 0.95) %>% # optional: use pca (reduction of features)
  #step_smote(all_outcomes(), neighbors=4) # K neighbors
  # OR step_upsample(all_outcomes()) OR step_downsample(all_outcomes())

rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(finalize(mtry(), traindat),
                            min_n(),
                            levels = 6) # grid of L^2 tuning possibilities

folds <- vfold_cv(traindat, v = 11, repeats =1) # K-folds

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

vroom_write(rf_submit,"amazonOutput.csv", delim = ",")


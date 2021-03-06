
######################################
## Importations des libraries ########
#####################################

library(tidyverse)
library(tidymodels)
library(kknn)

#########################  Mise en Activit� ####################

glimpse(diamonds)

#### S�lection variables num�riques

diamants <-
  diamonds %>%
  select_if(is.numeric) %>%
  slice(1:1000)

##### On divise nos donn�es en donn�es d'entrainement (70%) et de test (30%)

diamants_split <- initial_split(diamants, prop = 0.7)
diamants_train <- training(diamants_split)
diamants_test <- testing(diamants_split)

##### On commence par sp�cifier le mod�le, on dit qu'on veut un knn mais dont le
### k sera r�gl�

# Pour tidymodels, Pour les tidymodels, l'approche de sp�cification d'un mod�le se 
# veut plus unifi�e

# - Sp�cifiez le type de mod�le en fonction de sa structure math�matique (par exemple, 
# r�gression lin�aire, for�t al�atoire, K -plus proches voisins, etc.).

# - Sp�cifiez le moteur pour le montage du mod�le: Le plus souvent, cela refl�te 
#le package qui doit �tre utilis�.

# - Si n�cessaire, d�clarez le mode du mod�le Le mode refl�te le type de r�sultat 
#de pr�diction. Pour les r�sultats num�riques, le mode est la r�gression ; pour 
# les r�sultats qualitatifs, il s'agit de la classification

knn_spec <-  
  nearest_neighbor(
    mode = "regression",
    neighbors = tune()
  ) %>%
  set_engine("kknn")

knn_spec

### Ici, on cr�e l'objet de la mod�lisation knn_spec qui utilise la fonction de 
## d�claration d'objet nearest_neighbor avec un mode de regression et qui utilise 
#un neighbor qui sera r�gl� avec le package kknn

#### Preprocessing: On normalise ici nos donn�es

rec_diamants <-
  recipe(price ~ ., data = diamants_train) %>%
  step_scale(all_predictors())

rec_diamants

### C'est un objet qui applique sur tous les pr�dicteurs le scale


#### On fait le K-Cross Validation sur les donn�es de train

resamples <- vfold_cv(diamants_train, v = 10)
resamples$splits
#### ajouter un workflow: qui combine une specification
# de modele avec une specification d'une � recette � d'un pretraitement

wf_diamants <-
  workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(rec_diamants)


tune_res <-
  wf_diamants %>%
  tune_grid(
    resamples = resamples,
    grid = grid_regular(neighbors(range = c(1, 100)), levels = 51),
    metrics = metric_set(mae)
  )
#grid_regular: pr�cise les param�tres ici par bons du length(vec)/levels

res_cv<-collect_metrics(tune_res); res_cv


K_optimal <- select_best(tune_res) ; K_optimal# 24
tRMSE_optimal <- min(res_cv[["mean"]]); tRMSE_optimal # 49.46024


plot(
  x = res_cv[["neighbors"]],
  y = res_cv[["mean"]],
  type = "b",
  pch = 19,
  cex = 0.5,
  xlab = "Valeur de K",
  ylab = "tRMSE",
  main = "Resultats de la validation crois�e"
)
abline(v = K_optimal[["neighbors"]], lty = "dashed", col = "red")
points(
  x = K_optimal[["neighbors"]],
  y = res_cv[["mean"]][which(res_cv[["neighbors"]] == K_optimal[["neighbors"]])],
  col = "red", pch = 19, cex = 0.7
)

####  On r��ntraine notre knn sur tous nos donn�es d'entrainement

knn_spec_optimal <-
  nearest_neighbor(
    mode = "regression",
    neighbors = K_optimal[["neighbors"]] # On sp�ecifie le mod`ele avec le K optimal trouv�e
  ) %>%
  set_engine("kknn")

wf_diamants_optimal <- # On cr�ee un nouveau workflow avec la specification du modele optimal
  workflow() %>%
  add_model(knn_spec_optimal) %>%
  add_recipe(rec_diamants)

### On fit les donn�es sur les donn�es de train
knn_optimal_fit <- # On entraine le modele optimal sur toute la base de donnees d'entrainement
  wf_diamants_optimal %>%
  fit(data = diamants_train)

pred_test_knn <- predict(knn_optimal_fit, new_data = diamants_test)

knn_RMSE_test <- sqrt(mean((pred_test_knn[[".pred"]] - diamants_test[["price"]]) ^ 2)) # 72.70373

lm_spec <- linear_reg() %>% set_engine("lm")

lm_fit <- fit(lm_spec, formula = price ~ ., data = diamants_train)
pred_test_lm <- predict(lm_fit, new_data = diamants_test)
lm_RMSE_test <- sqrt(mean((pred_test_lm$.pred - diamants_test$price) ^ 2)) # 251.0236

tibble(
  price = diamants_test[["price"]],
  knn = pred_test_knn[[".pred"]],
  lm = pred_test_lm[[".pred"]]
) %>%
  head(6) %>%
  round()

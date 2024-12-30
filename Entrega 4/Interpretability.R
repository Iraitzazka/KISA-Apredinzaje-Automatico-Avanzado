# Feature Importance ------------------------------------------------------
library(iml)
library(caret) #instalalas previamente si es que no lo estaban
data("iris") # problema aburrido pero peque~no para poder analizarlo
summary(iris)
TrainData <- iris[,1:4]
TrainClasses <- iris[,5]
# aprendemos un modelo predictivo;
# que puede ser uno aprendido con la librera "caret";
# modelo "naive_bayes" se implementa en el paquete "naivebayes";
# listado de algoritmos de clasificacion ofrecidos por caret:
# https://topepo.github.io/caret/train-models-by-tag.html
nbFit <- train(TrainData, TrainClasses, "naive_bayes", tuneLength = 10,
               trcontrol=trainControl(method="cv"))
# creamos mediante "iml" un objeto que incluye el modelo aprendido para su analisis
model <- Predictor$new(nbFit, data=TrainData, y= TrainClasses)
# metrica de evaluacion -> "cross-entropy" de la clase real
imp <- FeatureImp$new(model, loss="ce")
# interpreta grafica y compara resultados numericos entre features en forma de "proporciones";
# no como valores absolutos
plot(imp)
imp$results

# ALE ------------------------------------------------------
ale <- FeatureEffect$new(model, feature="Petal.Length")
ale$plot()
# Para todas las predictoras
effs <- FeatureEffects$new(model)
# "patchwork" library is needed for plotting
library(patchwork)
plot(effs)

# Feature interactions ------------------------------------------------------

# 2-way interaction entre Petal.Length y el resto predictoras;
# se realiza para cada problema de clasificacion one-class versus rest-of-the-classes

interact <- Interaction$new(model, feature="Petal.Length")
plot(interact)

# Surrogate trees  ------------------------------------------------------

# aprendemos un multi-layer perceptron
mlpFit <- caret::train(TrainData, TrainClasses, "mlpML", tuneLength = 2)
model <- Predictor$new(mlpFit, data=TrainData, y= TrainClasses)
# aprendemos el surrogateTree que trata de explicar como clasifica el MLP
surrogateTree <- TreeSurrogate$new(model, maxdepth = 2)
plot(surrogateTree)

# SHAP  ------------------------------------------------------

library(kernelshap)
library(shapviz) #para visualizar los valores de "interpretability"
library(caret)
diabetes <- read.csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

colnames(diabetes)
diabetes$Outcome <- as.factor(diabetes$Outcome)
knnModel <- caret::train(Outcome ~ ., data=diabetes,method="knn",preProc=c("center","scale"))
# calculo SHAP values para 100 muestras aleatorias, aliviando c´omputo ´
subsample100diabetes <- diabetes[sample(nrow(diabetes),100), ]
knnshap <- kernelshap(knnModel, X=diabetes[,-9], bg_X= subsample100diabetes, type="prob")
knnSHAPViz <- shapviz(knnshap) # para visualizar con la librer´ıa "shapviz"
# SHAP values centrados en la "clase 1": prueba a quitar "[[1]]"
beeSHAPknn <- sv_importance(knnSHAPViz, kind="bee")[[1]]
beeSHAPknn
waterfallSHAPknn <- sv_waterfall(knnSHAPViz, row_id=50)[[1]]
waterfallSHAPknn$labels$title = "SHAP waterfall values"
waterfallSHAPknn


library("SHAPforxgboost")
diabetes <- read.csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
colnames(diabetes)
X1 = as.matrix(diabetes[,-9]) # dejar las predictoras
# aprender el modelo XGBoost. Fijar predictoras y clase


mod1 = xgboost::xgboost(
  data = X1, label = diabetes$Outcome, gamma = 0, eta = 1,
  lambda = 0,nrounds = 1, verbose = FALSE)
# calcular los SHAP values para cada caso - segun modelo aprendido
shap_values <- shap.values(xgb_model = mod1, X_train = X1)
shap_values$mean_shap_score
shap_values_diabetes <- shap_values$shap_score
shap_long_diabetes <- shap.prep(xgb_model = mod1, X_train = X1)
# **SHAP summary plot**
shap.plot.summary(shap_long_diabetes)
shap.plot.summary(shap_long_diabetes, x_bound = 1.5, dilute = 10)


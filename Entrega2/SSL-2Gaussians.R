set.seed(1)
library(RSSL)
df <- generate2ClassGaussian(2000,d=2,var=0.6)

# define the supervised classifier and SSL strategy; 
# and their parameters
# Clasificador rojo. Este usa solo 
classifiers <- list("LS"=function(X,y,X_u,y_u) {
  LeastSquaresClassifier(X,y,lambda=0)}, 
  "Self"=function(X,y,X_u,y_u) {
    SelfLearning(X,y,X_u,LeastSquaresClassifier)}
)
# Clasificador azul
classifiers2 <- list("SVM"=function(X,y,X_u,y_u) {
  SVM(X,y, C=1)}, 
  "Self"=function(X,y,X_u,y_u) {
    SelfLearning(X,y,X_u,SVM)}
)


# define the type of performance metric: 
# the vertical axe in the plotted curve
measures <- list("Accuracy" =  measure_accuracy)

# first line --> define unlabeled and labeled part of the data frame
# second line --> fix already defined classifier, SSL and metric
# third line --> 40% of the samples for testing and plotting the curves
# "fraction" of labeled objects varies from 0% to 100%: remaining ones, unlabeled
# repeated the process 10 times
lc1 <- LearningCurveSSL(as.matrix(df[,1:2]), df$Class,
                        classifiers=classifiers, measures=measures, 
                        type="fraction", test_fraction = 0.4, repeats=3)
#Type fraction implica que para el training, siempre uso 1200 datos, 
#pero va cambiando la fraccion de labeled points, de 0% labeled a 100% labeled (eje x)
#Esto se hace para los dos clasificadores, para el semisupervisado y el supervisado

#Tiene sentido que funcione mejor la semisupervision cuando no tengo muchas instancias etiquetadas.
# analyze where the green curve (SSL) is above the red curve (standard supervised)
plot(lc1)




epinions = read.csv(file = "http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/epinions5.arff.csv", header=TRUE, sep=",")
umic = read.csv(file = "http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/UMIC-SA-training.arff.csv", header=TRUE, sep=",")
spambase = read.csv(file = "http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/spambase.csv", header=TRUE, sep=",")
movie = read.csv(file="movie-polarity-10662reviews.csv", header=TRUE, sep=",")
airlines = read.csv(file="TweetsAirlines80terminos.csv", header=TRUE, sep=",")
covid

lc2 <- LearningCurveSSL(as.matrix(epinions[,2:1015]), as.factor(epinions$sentiment), 
                        classifiers=classifiers, measures=measures,
                        type="fraction", test_fraction=0.2, repeats = 3)

plot(lc2)

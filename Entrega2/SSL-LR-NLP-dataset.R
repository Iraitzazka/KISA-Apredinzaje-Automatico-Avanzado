set.seed(1)
# define the supervised classifier (logistic regression) and SSL strategy (self learning); 
# and their parameters

classifiers3 <- list("LogisticRegression"=function(X,y,X_u,y_u) {
  LogisticRegression(X,y, lambda=0)}, 
  "Self"=function(X,y,X_u,y_u) {
    SelfLearning(X,y,X_u,LogisticRegression)}
)

# define the type of performance metric: 
# the vertical axe in the plotted curve
measures <- list("Accuracy" =  measure_accuracy)

# first line --> define unlabeled and labeled part of the data frame
# second line --> fix already defined classifier, SSL and metric
# third line --> 40% of the samples for testing and plotting the curves
# "fraction" of labeled objects varies from 0% to 100%: remaining ones, unlabeled
# repeated the process 10 times

spambase = read.csv(file = "http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/spambase.csv", 
                    header=TRUE, sep=",")
lc2 <- LearningCurveSSL(as.matrix(spambase[,1:57]), as.factor(spambase$class), 
                        classifiers=classifiers3, measures=measures,
                        type="fraction", test_fraction=0.5, repeats = 10)

plot(lc2)

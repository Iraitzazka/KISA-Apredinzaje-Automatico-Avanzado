# csv file from
# https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts
# binary annotation: whether the text is considered anti-LGBT cyberbullying (1) or not (0)
library(tm)
lgtb = read.csv("anti-lgbt-cyberbullying.csv", stringsAsFactors = F, header=T)
lgtb_corpus = Corpus(VectorSource(lgtb$text))
lgtb_corpus = tm_map(lgtb_corpus, content_transformer(tolower))
lgtb_corpus = tm_map(lgtb_corpus, removeNumbers)
lgtb_corpus = tm_map(lgtb_corpus, removePunctuation)
lgtb_corpus = tm_map(lgtb_corpus, removeWords, stopwords("english"))
lgtb_corpus = tm_map(lgtb_corpus, stemDocument)
lgtb.dtm = DocumentTermMatrix(lgtb_corpus)
lgtb.dtm = removeSparseTerms(lgtb.dtm, 0.99)
dim(lgtb.dtm)

labels = lgtb$anti_lgbt
lgtb.dtm = cbind(lgtb.dtm,labels)
lgtb.dtm.matrix = as.matrix(lgtb.dtm)
colnames(lgtb.dtm.matrix)[277] = "label"

library(RSSL)
classifiers <- list("LeastSquaresClassifier"=function(X, y, X_u, y_u) {
LeastSquaresClassifier(X,y)},"Self"=function(X,y,X_u,y_u) {SelfLearning(X,y,X_u, method=LeastSquaresClassifier)})

measures <- list("Accuracy"=measure_accuracy)

lc2 <- LearningCurveSSL(lgtb.dtm.matrix[,1:276], as.factor(lgtb.dtm.matrix[,277]), 
                        classifiers=classifiers, measures=measures,
                        type="fraction", test_fraction=0.3, repeats = 5)
plot(lc2)

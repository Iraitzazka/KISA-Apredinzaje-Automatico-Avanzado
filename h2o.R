library(h2o)
h2o.init()
h2o.init(ip='localhost', port=54321, nthreads=-1, max_mem_size = '4g')

train_url_MNIST = "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_url_MNIST = "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train = h2o.importFile(train_url_MNIST)
test = h2o.importFile(test_url_MNIST)
#Nos aseguramos de que la variable clase (la ´ultima) sea de tipo factor.
Class = colnames(train)[length(train)]
train[[Class]] = as.factor(train[[Class]])
test[[Class]] = as.factor(test[[Class]])
Predictors = colnames(train)[1:length(train)-1]
#Entrenamos una red neuronal Multilayer Feed-Forward en el conjunto de train.
#Ojo con el n´umero de nodos por capa. En estas l´ıneas, 200 nodos por cada una de las dos capas.
#Si los tiempos de c´omputo son excesivos, red´ucelos.
#Cuando lo lances, abre el monitor del sistema y visualiza la carga de todas tus CPUs y memoria
deepLearning=h2o.deeplearning(x=Predictors, y=Class, training_frame = train,
                              hidden=c(200,200))
#Testeamos el modelo aprendido en el conjunto de test.
performance=h2o.performance(deepLearning,test)
#F´ıjate en las distintas m´etricas de evaluaci´on y matriz de confusi´on: qu´e elegante.
performance

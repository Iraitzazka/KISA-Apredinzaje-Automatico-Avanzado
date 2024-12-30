# Instalar y cargar el paquete Rtsne si no está instalado
if (!require(Rtsne)) install.packages("Rtsne", dependencies = TRUE)
library(Rtsne)

# Cargar el conjunto de datos iris (es un conjunto de datos de ejemplo común)
data(iris)
head(iris)

# Tomamos solo las características (sin la columna de etiquetas de especie)
data <- iris[, 1:4]

# Aplicar t-SNE (usando 2 dimensiones de salida)
set.seed(42)  # Para reproducibilidad
tsne_result <- Rtsne(data, dims = 2, pca = TRUE, perplexity = 30, check_duplicates = FALSE)
#Esta perplexity esta relacionada con la varianza de las distribuciones gaussianas sobre los puntos

# tsne_result$Y contiene las coordenadas de los puntos en 2D
tsne_data <- data.frame(tsne_result$Y)
colnames(tsne_data) <- c("V1", "V2")  # Nombrar las columnas de los resultados
tsne_data$Species <- iris$Species

# Graficar los resultados
library(ggplot2)

ggplot(tsne_data, aes(x = V1, y = V2, color = Species)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "Visualización de t-SNE", x = "Dim 1", y = "Dim 2") +
  theme(legend.position = "top")

rm(list = ls())
graphics.off()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

semilla <- 2026

# =====================================================================
# 1. Cargar librerías necesarias
# =====================================================================
library(readr)
library(caret)
library(naivebayes)
library(DataExplorer)
library(ggplot2)
library(reshape2)
library(tidyr)
library(dplyr)
library(pROC)        # [NUEVO] para la curva ROC y el AUC en test

# =====================================================================
# 2. Cargar el dataset Heart.csv
# =====================================================================
df <- read.csv("Heart.csv", sep = ";", header = TRUE)

cat("Dimensiones del dataset:\n")
print(dim(df))
str(df)
summary(df)
plot_missing(df)     # 0 % de faltantes en todas las variables -> no se imputa

# =====================================================================
# 3. Preparación y Transformación de Tipos de Datos
# =====================================================================
target <- "disease"
df[[target]] <- as.factor(df[[target]])
levels(df[[target]]) <- c("Sano", "Enfermo")

# [CORREGIDO] print("\n...") imprime el "\n" literal; se usa cat()
cat("\n--- DISTRIBUCIÓN DE CLASES (DISEASE) ---\n")
print(table(df$disease))
print(prop.table(table(df$disease)) * 100)
# Clases balanceadas (≈ 54 % / 46 %) -> NO se aplica técnica de balanceo

# Variables categóricas como factores
df$sex     <- as.factor(df$sex)
df$cp      <- as.factor(df$cp)
df$fbs     <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang   <- as.factor(df$exang)
df$slope   <- as.factor(df$slope)
df$ca      <- as.factor(df$ca)
df$thal    <- as.factor(df$thal)
cat_cols <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

# Variables continuas como numéricas
df$age      <- as.numeric(df$age)
df$trestbps <- as.numeric(df$trestbps)
df$chol     <- as.numeric(df$chol)
df$thalach  <- as.numeric(df$thalach)
df$oldpeak  <- as.numeric(df$oldpeak)
num_cols <- c("age", "trestbps", "chol", "thalach", "oldpeak")
str(df)

# Correlación entre numéricas (supuesto de independencia de Naive Bayes)
datos_num <- Filter(is.numeric, df)
descrCor <- cor(datos_num, use = "complete.obs")
print(round(descrCor, 3))   # todas |r| < 0.4 -> el supuesto es razonable

# =====================================================================
# EDA
# =====================================================================
plot_bar(df, title = "Distribución de Variables Categóricas", by = "disease")

df_long <- df %>%
  select(all_of(num_cols), disease) %>%
  pivot_longer(cols = -disease, names_to = "Variable", values_to = "Valor")

ggplot(df_long, aes(x = Valor, fill = as.factor(disease))) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribución de Variables Numéricas por Enfermedad",
       x = "Valor", y = "Frecuencia", fill = "Disease") +
  theme_minimal()

g_box <- ggplot(df_long, aes(x = disease, y = Valor, fill = disease)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.shape = 16, outlier.size = 2) +
  facet_wrap(~ Variable, scales = "free_y") +
  labs(title = "Boxplots de Variables Numéricas por Condición Cardíaca",
       subtitle = "Identificación de distribución y valores atípicos (outliers en rojo)",
       x = "Diagnóstico (Disease)", y = "Valor de la Variable") +
  scale_fill_manual(values = c("Sano" = "#3182bd", "Enfermo" = "#de2d26")) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(face = "bold", size = 10))
g_box

# =====================================================================
# 4. Partición Train / Test (Estratificada)
# =====================================================================
set.seed(semilla)
trainIndex <- createDataPartition(df[[target]], p = 0.7, list = FALSE)
train <- df[trainIndex, ]
test  <- df[-trainIndex, ]
print(paste("Observaciones en Train:", nrow(train), " | Test:", nrow(test)))

prop.table(table(train$disease)) * 100
prop.table(table(test$disease)) * 100

# =====================================================================
# [NUEVO] 5. Detección de outliers (IQR y Z-Score, solo en TRAIN)
# =====================================================================
cat("\n--- OUTLIERS EN TRAIN ---\n")
for (v in num_cols) {
  x   <- train[[v]]
  q   <- quantile(x, c(0.25, 0.75)); iqr <- q[2] - q[1]
  n_iqr <- sum(x < q[1] - 1.5 * iqr | x > q[2] + 1.5 * iqr)
  n_z   <- sum(abs(scale(x)) > 3)
  cat(sprintf("%-9s IQR: %2d | Z-Score: %2d\n", v, n_iqr, n_z))
}
# Decisión: se MANTIENEN. Son valores clínicamente posibles (p.ej. chol = 564)
# y Naive Bayes con kernel es poco sensible a valores extremos aislados.

# =====================================================================
# 6. Validación cruzada
# =====================================================================
# [CORREGIDO] classProbs + twoClassSummary para elegir el modelo por ROC (AUC)
# y no por Accuracy (en clínica importa no dejar enfermos sin detectar)
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# =====================================================================
# 7. Entrenamiento del modelo
# =====================================================================
# [CORREGIDO] ERROR PRINCIPAL:
#   Con la fórmula (disease ~ .) caret convierte los factores en dummies 0/1
#   y con preProcess = center/scale las trata como numéricas. Naive Bayes
#   dejaba de usar las variables categóricas (por eso laplace no cambiaba nada).
#   -> Se usa la interfaz x / y y se elimina preProcess
#      (knnImpute sobraba: no hay faltantes; center/scale no los necesita NB).
predictores <- setdiff(names(train), target)

set.seed(semilla)
modelo_nb <- train(
  x = train[, predictores],
  y = train[[target]],
  method = "naive_bayes",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = expand.grid(
    laplace   = c(0.5, 1, 2),       # mismo grid que el código del profesor
    usekernel = c(TRUE, FALSE),
    adjust    = c(0.5, 1, 1.5)
  )
)

# --- RESULTADOS DE VALIDACIÓN CRUZADA ---
print(modelo_nb)            # ya NO debe aparecer "Pre-processing: centered..."
print(modelo_nb$bestTune)
plot(modelo_nb, main = "Rendimiento del Modelo según Hiperparámetros (Naive Bayes)")

# [NUEVO] Distribuciones aprendidas por variable (como en el código del profesor)
print(get_cond_dist(modelo_nb$finalModel))   # Gaussian / KDE / Categorical
print(modelo_nb$finalModel)                  # tablas P(variable | clase)

# =====================================================================
# 8. Evaluación final en TEST
# =====================================================================
pred_test <- predict(modelo_nb, newdata = test[, predictores])
cm_test <- confusionMatrix(pred_test, test[[target]], positive = "Enfermo")
print(cm_test)

# [NUEVO] Probabilidades, curva ROC y AUC
prob_test <- predict(modelo_nb, newdata = test[, predictores], type = "prob")
roc_test  <- roc(test[[target]], prob_test$Enfermo, levels = c("Sano", "Enfermo"))
cat("\nAUC en Test:", round(auc(roc_test), 4), "\n")
plot(roc_test, main = "Curva ROC - Naive Bayes (Test)", print.auc = TRUE)

# [NUEVO] Ajuste del umbral: bajar el umbral sube la sensibilidad
cat("\n--- SENSIBILIDAD / ESPECIFICIDAD SEGÚN UMBRAL ---\n")
for (u in c(0.3, 0.4, 0.5)) {
  pred_u <- factor(ifelse(prob_test$Enfermo >= u, "Enfermo", "Sano"),
                   levels = c("Sano", "Enfermo"))
  cm_u <- confusionMatrix(pred_u, test[[target]], positive = "Enfermo")
  cat(sprintf("Umbral %.1f | Sensibilidad: %.3f | Especificidad: %.3f | Accuracy: %.3f\n",
              u, cm_u$byClass["Sensitivity"], cm_u$byClass["Specificity"],
              cm_u$overall["Accuracy"]))
}


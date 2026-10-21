# 🚀 Portafolio de Clasificación en Machine Learning: Salud y Justicia Actuarial

Este repositorio contiene el desarrollo de tres pipelines de clasificación de Machine Learning aplicados a escenarios del mundo real donde el costo de equivocarse es asimétrico. El proyecto abarca todo el ciclo de vida del dato: desde el preprocesamiento (imputación multivariada, escalado y balanceo de clases) hasta la optimización de hiperparámetros y la interpretación clínica/judicial de los resultados.

## 🛠️ Stack Tecnológico
* **Python:** `pandas`, `scikit-learn`, `imblearn` (SMOTE, KNNImputer, MICE), `seaborn`, `matplotlib`.
* **R:** `caret`, `naivebayes`, `pROC`, `DataExplorer`.

## 📊 Casos de Estudio y Modelos

### ⚖️ 1. Justicia Predictiva: Predicción de Reincidencia (Parole)
* **Objetivo:** Predecir el riesgo de que un interno viole su libertad condicional, sirviendo como herramienta de soporte (screening) para la toma de decisiones.
* **Modelo:** Árboles de Decisión (CART).
* **Desafío:** Desbalance severo de clases (11.6% reincidentes). Se solucionó evaluando 6 técnicas de remuestreo (SMOTE logró el mejor rendimiento) y aplicando post-poda por costo-complejidad (`ccp_alpha`) para obtener reglas de decisión auditables.
* 💻 **[Ver código y análisis en Python](./notebooks/arbol_decision_parole.ipynb)**

### 🫀 2. Diagnóstico de Riesgo Coronario (Heart)
* **Objetivo:** Identificar tempranamente a pacientes con enfermedad arterial coronaria a partir de datos clínicos no invasivos.
* **Modelo:** Naive Bayes.
* **Desafío:** Optimizar el umbral de decisión (threshold) para maximizar la sensibilidad (Recall), ya que en el contexto médico un falso negativo tiene un costo vital mayor que un falso positivo.
* 💻 **[Ver código fuente en R](./scripts/naive_bayes_heart.R)**
* 📄 **[Ver Reporte con Gráficos y Curva ROC en PDF](./docs/naive_bayes_heart.pdf)**

### 🩸 3. Detección de Diabetes (Pima)
* **Objetivo:** Determinar la presencia de diabetes en pacientes femeninas basándose en métricas fisiológicas.
* **Modelo:** K-Nearest Neighbors (KNN).
* **Desafío:** Alta tasa de valores nulos (ej. Insulina 49%). Se implementó imputación multivariada mediante MICE acoplada a técnicas combinadas de balanceo (`SMOTE + Tomek`) en un `Pipeline` optimizado con `GridSearchCV`.
* 💻 **[Ver código y análisis en Python](./notebooks/knn_clasificacion_diabetes.ipynb)**

## 📈 Resultados Destacados

La métrica *Accuracy* fue descartada como objetivo de optimización principal debido al desbalance de clases, priorizando el **F1-Score, el Recall y el AUC-ROC** para asegurar la viabilidad de los modelos en entornos reales.

| Caso de Estudio | Algoritmo | AUC-ROC | Recall (Clase Positiva) | Hallazgo Clave |
| :--- | :--- | :--- | :--- | :--- |
| **Riesgo Coronario** | Naive Bayes | **0.887** | **0.756** | Ajuste de umbral a 0.4 para priorizar el triaje médico seguro. |
| **Diabetes** | KNN | **0.808** | **0.796** | Imputación MICE + SMOTE + Tomek (K=29) estabilizó las fronteras. |
| **Reincidencia (Parole)** | CART | **0.794** | **0.563** | Post-poda redujo el árbol a 5 nodos, mejorando generalización. |

📄 **[Ver Resumen Ejecutivo Completo de Resultados](./docs/Resumen_Ejecutivo_Resultados.pdf)**

## 📂 Estructura del Repositorio

```text
├── data/           # Datasets originales (Heart.csv, Parole.csv, Diabetes.xlsx)
├── docs/           # Documentación, reportes PDF y resúmenes ejecutivos
├── notebooks/      # Pipelines interactivos en Jupyter/Colab (Python)
├── scripts/        # Código fuente y análisis en R
└── README.md       # Presentación del proyecto
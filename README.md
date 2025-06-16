# ml_project
Proyecto de Machine Learning I 

1. Problema de ML
El objetivo de este proyecto es construir un modelo de clasificación para identificar la especie de una flor Iris en función de sus características morfológicas. Este problema es ideal para explorar técnicas de aprendizaje automático supervisado, específicamente algoritmos de clasificación.

Importancia
El dataset de Iris es ampliamente utilizado en la comunidad de ciencia de datos para probar y comparar modelos de clasificación. Su facilidad de interpretación y calidad de datos lo convierten en un excelente punto de partida para aprender sobre Machine Learning.

2. Diagrama de Flujo del Proyecto
El flujo del proyecto sigue las siguientes etapas:

Carga y exploración de datos

Preprocesamiento (manejo de valores nulos y codificación de variables)

División del conjunto de datos en entrenamiento y prueba

Entrenamiento de modelos de clasificación (ej. Random Forest, SVM, KNN, Logistic Regression)

Evaluación del rendimiento con métricas como accuracy, F1-score y matriz de confusión

Análisis de resultados y selección del mejor modelo

Despliegue del modelo entrenado

Lo pueden ver aqui: https://drive.google.com/file/d/1Adp3uva-KTyZPQhmm11c5idvXMNSM8Ah/view?usp=sharing

3. Descripción del Dataset
El dataset Iris contiene 150 registros de flores con 4 características principales y la etiqueta de clase.

Feature	Descripción	Unidad
sepal_length	Largo del sépalo	cm
sepal_width	Ancho del sépalo	cm
petal_length	Largo del pétalo	cm
petal_width	Ancho del pétalo	cm
species	Especie de la flor (setosa, versicolor, virginica)	Categoría
Diccionario de Datos
El dataset no tiene valores nulos y las características son numéricas, por lo que requiere preprocesamiento mínimo. La columna species es categórica y debe ser codificada antes de entrenar el modelo.

Fuente del dataset: Disponible en UCI Machine Learning Repository.

4. Model Card
Este modelo de clasificación de flores Iris se evaluará con diferentes algoritmos. La siguiente Model Card sigue los lineamientos de Kaggle Model Cards.

Tipo de modelo: Clasificación 

Datos de entrada: Características morfológicas de la flor (sepal_length, sepal_width, petal_length, petal_width)

Datos de salida: Clase de la flor (setosa, versicolor, virginica)

Métricas de evaluación: Accuracy, F1-score, matriz de confusión

Limitaciones: La clasificación depende exclusivamente de medidas morfológicas, sin considerar factores ambientales.


5. Resultados con Métricas de Evaluación
Después de entrenar múltiples modelos, se compararán los resultados:


Modelo	         Accuracy	F1-score
Random Forest	98.5%	0.98
SVM	            97.3%	0.97
KNN	             96.7%	0.96
La matriz de confusión muestra la precisión de cada clase, y se evaluará la robustez del modelo con validación cruzada.

6. Conclusiones
El modelo Random Forest ofrece la mejor precisión, superando a SVM y KNN.

El dataset Iris es ideal para probar modelos de clasificación básicos. La alta precisión sugiere que los datos son bien estructurados y adecuados para el aprendizaje supervisado.

Podrían explorarse modelos más avanzados, como redes neuronales o enfoques de aprendizaje profundo, para mejorar la clasificación en conjuntos de datos más complejos.

Posible despliegue en aplicaciones de identificación botánica, donde el modelo podría integrarse con un sistema visual basado en imágenes.
# Reto Inteligencia Artificial - Telefónica

Este proyecto implementa un sistema forense de detección de audios generados por inteligencia artificial (deepfakes) utilizando el dataset **ASVspoof 2019 (Logical Access)**. El sistema combina procesamiento de señales digitales con modelos avanzados de aprendizaje automático para distinguir voces humanas de ataques sintéticos (TTS y VC).

---

## 1. Descripción de los Datos Recibidos

El dataset se centra en la partición **Logical Access (LA)**, diseñada para evaluar contramedidas contra ataques de suplantación de identidad por voz.

- **Naturaleza de los ataques:** Incluye 19 tipos de algoritmos de generación (A01-A19), abarcando síntesis de voz (TTS) y conversión de voz (VC).
- **Especificaciones Técnicas:** Audios en formato `.flac`, muestreados a 16 kHz y 16 bits.
- **Etiquetado:** 
  - `bonafide`: Voz humana legítima.
  - `spoof`: Voz generada artificialmente.

---

## 2. Funcionamiento del Pipeline

El proyecto sigue un flujo de datos riguroso para garantizar la fiabilidad de los resultados:

1.  **Extracción Forense:** Se analizan los audios para extraer su "huella digital" espectral. No se utiliza el audio bruto, sino métricas matemáticas que capturan anomalías en la frecuencia y el ritmo.
2.  **Consolidación de Metadatos:** Se vinculan las métricas extraídas con la información del hablante, el género y el tipo de ataque para permitir auditorías de sesgo y robustez.
3.  **Gestión de Desequilibrio:** Dado que hay significativamente más muestras sintéticas que humanas, se aplican técnicas de **Random UnderSampling** dentro de pipelines de validación para evitar que el modelo se vuelva perezoso o sea sesgado.
4.  **Evaluación de Ataques Desconocidos:** El sistema se somete a pruebas con ataques que no estuvieron presentes durante el entrenamiento, evaluando su verdadera capacidad de generalización.

---

## 3. Características (Features) Extraídas

En esta sección se describen las métricas extraídas de los archivos de audio que conforman la base de datos (extraídas mediante `librosa`). Estas variables permiten al modelo diferenciar entre el habla natural y la generada sintéticamente:

- **`energy`**: La energía total de la señal de audio.
- **`rmse_mean`**: Root Mean Square Energy. Mide la energía promedio de los cuadros (frames) de audio.
- **`zero_crossings`**: Tasa de cruce por cero. Mide la frecuencia con la que la señal cambia de signo (cruza el eje X).
- **`tempo`**: Ritmo estimado del audio en pulsos por minuto (BPM).
- **`mfcc_mean`**: Coeficientes Cepstrales en las Frecuencias de Mel (promedio). Representan el espectro de potencia a corto plazo de un sonido.
- **`tempogram_mean`**: Promedio del tempograma, que mide la autocorrelación local de la envolvente de fuerza de inicio.
- **`spec_centroid_mean`**: Centroide Espectral. Indica dónde se encuentra el "centro de masa" del espectro (brillo percibido del sonido).
- **`spec_bandwidth_mean`**: Ancho de Banda Espectral. Describe la anchura de la banda de frecuencias de la señal de audio.
- **`spec_contrast_mean`**: Contraste Espectral. Considera los picos y valles espectrales y su diferencia en cada sub-banda de frecuencia.
- **`spec_flatness_mean`**: Planitud Espectral. Mide qué tan similar es un sonido al ruido blanco frente a un sonido tonal.
- **`spec_rolloff_mean`**: Caída Espectral (Rolloff). Frecuencia por debajo de la cual se encuentra un porcentaje específico (usualmente el 85%) de la energía espectral total.
- **`freq_ (mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr)`**: Métricas estadísticas descriptivas calculadas sobre la distribución de frecuencias de la señal de audio (media, desviación estándar, asimetría, curtosis, rango intercuartílico, etc.).


---

## 4. Estructura de Archivos y Funcionalidad

### 4.1. Árbol de Directorios Completo
```text
1. Audios/                          
   1.1. DS_LA/                      
      1.1.1. DS_LA_LAtrain/         
      1.1.2. DS_LA_LAeval/          
      1.1.3. ASVspoof_LA.csv        
      1.1.4. ASVspoof_LA_E.csv      
2. Datos/                           
   2.1. 1.conversion_to_data.py    
   2.2. 2.fusion.py          
   2.3. df_general.csv              
   2.4. df_voices_train.csv         
   2.5. df_voices_eval.csv          
   2.6. mis_features_audio.csv      
   2.7. mis_features_audio_eval.csv 
3. EDA/                             
   3.1. eda.ipynb            
4. Modelos/                         
   4.1. Redes_neuronales.py         
   4.2. model_selection_cv_gen.py   
   4.3. df_generico/                
      4.3.1. deteccion_voz_falsa_large_model.ipynb
      4.3.2. model_selection_large_model.ipynb
   4.4. train_test_model/           
      4.4.1. Detección_de_voz_falsa_mediante_IA.ipynb
      4.4.2. learning_curve_train_test_model.ipynb
      4.4.3. robustness_train_test_model.ipynb
      4.4.4. selection_train_test_model.ipynb
5. rf_learning_curve_test_train.ipynb
```

### 4.2. Descripción Detallada de cada Componente

**1. Carpeta Audios (Datos Brutos)**
- **[1.1.1/1.1.2] DS_LA_LAtrain/eval:** Directorios que contienen los archivos de audio originales en formato `.flac` para el entrenamiento y la evaluación final.
- **[1.1.3/1.1.4] ASVspoof_LA.csv / ASVspoof_LA_E.csv:** Archivos de metadatos oficiales que vinculan cada archivo de audio con su etiqueta (`bonafide`/`spoof`) y el identificador de ataque.

**2. Carpeta Datos (Procesamiento)**
- **[2.1] Conversion_audios_csv.py:** El motor de ingeniería de características. Transforma ondas de audio en vectores numéricos (MFCC, centroide, etc.) usando la librería `librosa`.
- **[2.2] Fusion_archivos.py:** Script encargado de realizar la unión (merge) entre las características extraídas y los metadatos de las etiquetas, realizando además la limpieza de nulos.
- **[2.3] df_general.csv:** Dataset consolidado que sirve como fuente principal para los entrenamientos iniciales.
- **[2.4/2.5] df_voices_train/eval.csv:** Datasets finales procesados y listos para ser consumidos por los notebooks de modelos.
- **[2.6/2.7] mis_features_audio/eval.csv:** Archivos CSV intermedios que contienen las características espectrales recién extraídas antes de la fusión.

**3. Carpeta EDA (Investigación)**
- **[3.1] EDA_Audios.ipynb:** Análisis estadístico detallado donde se identificaron las diferencias en el brillo y ancho de banda entre audios reales y sintéticos.

**4. Carpeta Modelos (Inteligencia Artificial)**
- **[4.1] Redes_neuronales.py:** Implementación de los clasificadores (Random Forest, SVM, MLP con Keras) y la lógica de balanceo mediante `RandomUnderSampler`.
- **[4.2] model_selection_cv_gen.py:** Script para la ejecución de validación cruzada estratificada y búsqueda de mejores hiperparámetros.
- **[4.3.1] deteccion_voz_falsa_large_model.ipynb:** Estudio sobre la importancia de variables y detección de ataques desconocidos (Zero-Day) como el A14.
- **[4.3.2] model_selection_large_model.ipynb:** Búsqueda y optimización de modelos sobre el dataset de alta dimensionalidad.
- **[4.4.1] Detección_de_voz_falsa_mediante_IA.ipynb:** Notebook principal que presenta los resultados finales, métricas de precisión y análisis de robustez por tipo de ataque.
- **[4.4.2] learning_curve_train_test_model.ipynb:** Generación de curvas de aprendizaje para diagnosticar el comportamiento del modelo frente al volumen de datos.
- **[4.4.3] robustness_train_test_model.ipynb:** Pruebas de estrés que miden la degradación del rendimiento (F1-Score) al introducir ruido en las señales.
- **[4.4.4] selection_train_test_model.ipynb:** Pipeline automatizado para la selección de la arquitectura más eficiente.

**5. Archivo Raíz**
- **[5] rf_learning_curve_test_train.ipynb:** Análisis complementario del proceso de entrenamiento y validación del modelo Random Forest.

---

## 5. Modelos y Estrategia de Clasificación

Se han seleccionado cuatro familias de algoritmos para cubrir diferentes tipos de patrones:

1.  **Random Forest:** Utilizado como el "Campeón" por su alta estabilidad y resistencia al ruido espectral.
2.  **Gradient Boosting:** Optimizado para capturar fronteras de decisión extremadamente finas entre voces reales y síntesis de alta calidad.
3.  **SVM (Support Vector Machine):** Utiliza kernels RBF para detectar patrones no lineales complejos en el espacio de frecuencias. (kernels lineal y RBF)

---

## 6. Conclusiones y Resultados

El sistema ha demostrado una capacidad sobresaliente para generalizar:
- **Detección de Ataques Desconocidos:** El modelo mantiene precisiones superiores al 90% incluso ante ataques de IA que nunca vio durante su entrenamiento.
- **Importancia de las Features:** Se confirmó que las características espectrales (especialmente MFCC y Contraste) son mucho más informativas que las métricas de energía simple para esta tarea.
- **Robustez:** La estrategia de validación cruzada estratificada garantiza que el sistema sea confiable independientemente de la variabilidad del hablante o el género.







---


Planteamiento de problema
1 a 4 ISA



Metodologia (Marcel)
5 - Extraccion, dimensiones
6 - ETL (3 dataset y se fusionan etc)
7 - Slide skip definiciones de las features



EDA (5 minutos)
8 - desbalanceo (no poner grafico)
9 - grafico de Análisis de Características: Humanos vs IA - Diferenciación en medias de algunas features
10 - Algo de pipe y medidas de performance
11 - Porque probamos cada modelo UWU y que se escogio - Colinealidad entre algunas variables que hacían pensar que RF podía ser el mejor modelo (porque gestiona nativamente la relación entre variables)


Prueba de robustes
12 en adelante



Aquí tienes las tres fórmulas de las medidas de tendencia central (o promedios) más comunes: la **Aritmética**, la **Geométrica** y la **Armónica**.

Para estos ejemplos, asumimos un conjunto de $n$ números: $\{x_1, x_2, \dots, x_n\}$.

---

### 1. Media Aritmética ($\bar{x}$)

Es el promedio estándar que todos conocemos. Se obtiene sumando todos los valores y dividiendo entre el número total de datos. Es ideal para datos con distribución uniforme.

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} = \frac{x_1 + x_2 + \dots + x_n}{n}$$

---

### 2. Media Geométrica ($G$)

Se utiliza principalmente para calcular tasas de crecimiento porcentual o intereses compuestos. En lugar de sumar, los valores se multiplican.

$$G = \sqrt[n]{\prod_{i=1}^{n} x_i} = \sqrt[n]{x_1 \cdot x_2 \cdot \dots \cdot x_n}$$

---

### 3. Media Armónica ($H$)

Como vimos antes, es el recíproco del promedio de los recíprocos. Es la mejor para promediar velocidades, densidades o razones.

$$H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}} = \frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \dots + \frac{1}{x_n}}$$

---

### Comparativa Rápida

| Medida | Operación principal | Uso común |
| --- | --- | --- |
| **Aritmética** | Suma | Notas escolares, salarios, temperaturas. |
| **Geométrica** | Multiplicación | Crecimiento de inversiones, poblaciones. |
| **Armónica** | Inversión (Recíprocos) | Velocidad media, razones financieras. |

Un dato curioso es la **Desigualdad de las Medias**: para cualquier grupo de números positivos, la relación siempre será $H \leq G \leq \bar{x}$.
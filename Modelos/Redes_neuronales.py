import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Cargar y preparar datos
print("--- CARGANDO Y PREPARANDO DATOS ---")
df = pd.read_csv('./Datos/df_voicesS.csv')

# Crear variable objetivo (y)
df['y'] = df['Key'].map({'bonafide': 0, 'spoof': 1})

# Separar características (X) y objetivo (y)
drop_columns = ['Key', 'file_name', 'User_ID', 'Spoofing_ID', 'y']
X = df.drop(columns=drop_columns, errors='ignore')
y = df['y']

# Escalar los datos (importante para todos los modelos, vital para Redes Neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras. Test: {X_test.shape[0]} muestras.\n")

# Diccionario para guardar predicciones y comparar luego
predictions = {}

# ==========================================
# MODELO CLÁSICO 1: Random Forest
# ==========================================
print("--- ENTRENANDO RANDOM FOREST ---")
for n_trees in [50, 100]:
    print(f"  -> Entrenando Random Forest con {n_trees} árboles...")
    rf_model = RandomForestClassifier(
                                     n_estimators=n_trees,  # Número de árboles a utilizar. 
                                     random_state=42,   # Valor constante para asegurar resultados repetibles
                                     n_jobs=-1          # Número de núcleos de CPU a utilizar
                                     )
    rf_model.fit(X_train, y_train)
    predictions[f'Random Forest ({n_trees} árboles)'] = rf_model.predict(X_test)

# ==========================================
# MODELO CLÁSICO 2: Gradient Boosting
# ==========================================
print("--- ENTRENANDO GRADIENT BOOSTING ---")
for n_trees in [50, 100]:
    print(f"  -> Entrenando Gradient Boosting con {n_trees} árboles...")
    gb_model = GradientBoostingClassifier(
                                        n_estimators=n_trees, 
                                        random_state=42
                                        )
    gb_model.fit(X_train, y_train)
    predictions[f'Gradient Boosting ({n_trees} árboles)'] = gb_model.predict(X_test)

# ==========================================
# MODELO CLÁSICO 3: Support Vector Machine (SVM)
# ==========================================
from sklearn.svm import SVC
print("--- ENTRENANDO SUPPORT VECTOR MACHINE (SVM) ---")
for kernel in ['linear', 'rbf']:
    print(f"  -> Entrenando SVM con kernel '{kernel}'...")
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train)
    predictions[f'SVM ({kernel} kernel)'] = svm_model.predict(X_test)


# ==========================================
# MODELO DEEP LEARNING: Multi-Layer Perceptron (MLP)
# ==========================================
print("--- ENTRENANDO RED NEURONAL (MLP) ---")
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar en modo silencioso (verbose=0) para no saturar la consola
mlp_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
predictions['Red Neuronal (MLP)'] = (mlp_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()


# ==========================================
# EVALUACIÓN Y COMPARACIÓN
# ==========================================
print("==========================================")
print("REPORTE DE CLASIFICACIÓN (Comparativa)")
print("==========================================\n")

for model_name, y_pred in predictions.items():
    print(f"---------- {model_name} ----------")
    print(classification_report(y_test, y_pred, target_names=['Human (0)', 'IA (1)']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
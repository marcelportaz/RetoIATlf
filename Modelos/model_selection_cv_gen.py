import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================

print("--- CARGANDO Y PREPARANDO DATOS ---")

df_gen = pd.read_csv('./Datos/df_general.csv')

# Crear variable objetivo
df_gen['y'] = df_gen['Key'].map({
    'bonafide': 0,
    'spoof': 1
})

# Separar X e y
drop_columns = ['Key', 'file_name', 'User_ID', 'Spoofing_ID', 'y']

X = df_gen.drop(columns=drop_columns, errors='ignore')
y = df_gen['y']

print(f"Dataset cargado: {X.shape[0]} muestras y {X.shape[1]} variables\n")

# ==========================================
# 2. CONFIGURACIÓN DE CROSS VALIDATION
# ==========================================

# StratifiedKFold mantiene el balance de clases en cada fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

spoofing_ids = df_gen['Spoofing_ID']

# Métricas a evaluar
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# ==========================================
# 3. DEFINICIÓN DE MODELOS
# ==========================================

models = {

    # ---------------- RANDOM FOREST ----------------
    'Random Forest (50 árboles)': RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),

    'Random Forest (100 árboles)': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    'Random Forest (150 árboles)': RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    'Random Forest (200 árboles)': RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    'Random Forest (250 árboles)': RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1
    ),

    'Random Forest (300 árboles)': RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    # ---------------- GRADIENT BOOSTING ----------------
    'Gradient Boosting (50 árboles)': GradientBoostingClassifier(
        n_estimators=50,
        random_state=42
    ),

    'Gradient Boosting (100 árboles)': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    # ---------------- SVM ----------------
    'SVM (linear kernel)': SVC(
        kernel='linear',
        random_state=42
    ),

    'SVM (rbf kernel)': SVC(
        kernel='rbf',
        random_state=42
    )
}

# ==========================================
# 4. EVALUACIÓN CON CROSS VALIDATION
# ==========================================

print("==========================================")
print("EVALUACIÓN CON CROSS VALIDATION")
print("==========================================\n")

results = []

for model_name, model in models.items():

    print(f"--- Evaluando: {model_name} ---")

    # Pipeline:
    # 1. Escalado
    # 2. Undersampling SOLO en train de cada fold
    # 3. Modelo
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('undersampling', RandomUnderSampler(random_state=42)),
        ('model', model)
    ])

    # Cross Validation
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        groups=spoofing_ids,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    # Promedios
    accuracy_mean = np.mean(cv_results['test_accuracy'])
    precision_mean = np.mean(cv_results['test_precision'])
    recall_mean = np.mean(cv_results['test_recall'])
    f1_mean = np.mean(cv_results['test_f1'])

    # Desviación estándar
    accuracy_std = np.std(cv_results['test_accuracy'])
    f1_std = np.std(cv_results['test_f1'])

    results.append({
        'Modelo': model_name,
        'Accuracy': accuracy_mean,
        'Precision': precision_mean,
        'Recall': recall_mean,
        'F1-Score': f1_mean,
        'Std Accuracy': accuracy_std,
        'Std F1': f1_std
    })

    print(f"Accuracy : {accuracy_mean:.4f} (+/- {accuracy_std:.4f})")
    print(f"Precision: {precision_mean:.4f}")
    print(f"Recall   : {recall_mean:.4f}")
    print(f"F1-Score : {f1_mean:.4f}")
    print()

# ==========================================
# 5. TABLA FINAL COMPARATIVA
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='F1-Score',
    ascending=False
)

print("==========================================")
print("RESULTADOS FINALES")
print("==========================================")

print(results_df.to_string(index=False))
print()
# Quiero hacer un merge del csv de mis_features_audio.csv con el csv con ASVspoofing.csv, la columna filename, coincide con la columna archivo del csv de mis_features_audio.csv, quiero que el resultado se guarde en un nuevo csv llamado mis_features_fusionadas.csv
import pandas as pd
# Cargar ambos CSVs
# Asegúrate de que las rutas a los archivos CSV sean correctas
df_features_audio = pd.read_csv('mis_features_audio.csv')
df_asvspoofing = pd.read_csv('ASVspoofing.csv')
# Realizar el merge usando la columna 'archivo' de df_features_audio y 'filename' de df_asvspoofing
df_fusionado = pd.merge(df_features_audio, df_asvspoofing, left_on='archivo', right_on='filename', how='inner')
# Guardar el DataFrame fusionado en un nuevo CSV
df_fusionado.to_csv('mis_features_fusionadas.csv', index=False)
print("¡Fusión completada! El archivo 'mis_features_fusionadas.csv' ha sido creado con éxito.")   
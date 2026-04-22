import pandas as pd

df_features_audio = pd.read_csv('mis_features_audio.csv')
df_asvspoofing = pd.read_csv('Audios/DS_LA/ASVspoofing_LA.csv')
# Realizar el merge usando la columna 'file_name' de df_features_audio y 'File_name' de df_asvspoofing
df_fusionado = pd.merge(df_features_audio, df_asvspoofing, left_on='file_name', right_on='File_name', how='inner')
# Guardar el DataFrame fusionado en un nuevo CSV
df_fusionado.to_csv('mis_features_fusionadas.csv', index=False)
print("¡Fusión completada! El archivo 'mis_features_fusionadas.csv' ha sido creado con éxito.")   
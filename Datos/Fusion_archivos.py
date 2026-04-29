import pandas as pd

df_features_audio = pd.read_csv('./Datos/mis_features_audio.csv')
df_asvspoof = pd.read_csv('./Audios/DS_LA/ASVspoof_LA.csv')
# Realizar el merge usando la columna 'file_name' de df_features_audio y 'File_Name' de df_asvspoof
df_fusionado = pd.merge(df_features_audio, df_asvspoof, left_on='file_name', right_on='File_Name', how='inner')
# Eliminamos  la columna 'File_Name' para no contar con duplicados
df_fusionado.drop(columns=['File_Name'], inplace=True)
# Guardar el DataFrame fusionado en un nuevo CSV
df_fusionado.to_csv('./Datos/df_voices.csv', index=False)
print("¡Fusión completada! El archivo 'df_voices.csv' ha sido creado con éxito.")
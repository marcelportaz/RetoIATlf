import pandas as pd

# ==========================================
# Fusion df_voice_train
# ==========================================

df_features_audio = pd.read_csv('./Datos/mis_features_audio.csv')
df_asvspoof = pd.read_csv('./Audios/DS_LA/ASVspoof_LA.csv')
# Realizar el merge usando la columna 'file_name' de df_features_audio y 'File_Name' de df_asvspoof
df_fusionado = pd.merge(df_features_audio, df_asvspoof, left_on='file_name', right_on='File_Name', how='inner')
# Eliminamos  la columna 'File_Name' para no contar con duplicados
df_fusionado.drop(columns=['File_Name'], inplace=True)
# Guardar el DataFrame fusionado en un nuevo CSV
df_fusionado.to_csv('./Datos/df_voices_train.csv', index=False)
print("¡Fusión completada! El archivo 'df_voices_train.csv' ha sido creado con éxito.")

# ==========================================
# Fusion df_voice_train_eva
# ==========================================

df_features_audio = pd.read_csv('./Datos/mis_features_audio_eval.csv')
df_asvspoof = pd.read_csv('./Audios/DS_LA/ASVspoof_LA_E.csv')
# Realizar el merge usando la columna 'file_name' de df_features_audio y 'File_Name' de df_asvspoof
df_fusionado = pd.merge(df_features_audio, df_asvspoof, left_on='file_name', right_on='File_Name', how='inner')
# Eliminamos  la columna 'File_Name' para no contar con duplicados
df_fusionado.drop(columns=['File_Name'], inplace=True)
# Guardar el DataFrame fusionado en un nuevo CSV
df_fusionado.to_csv('./Datos/df_voices_eval.csv', index=False)
print("¡Fusión completada! El archivo 'df_voices_eval.csv' ha sido creado con éxito.")

# ==========================================
# Fusion df_voice_train y df_voice_eval
# ==========================================
df1 = pd.read_csv('./Datos/df_voices_train.csv')
df2 = pd.read_csv('./Datos/df_voices_eval.csv')

df_generico = pd.concat([df1, df2], ignore_index=True)

#get_dummies para la columna Gender
df_generico = pd.get_dummies(df_generico, columns=['Gender'])

# Guardar el resultado en un nuevo CSV
df_generico.to_csv('./Datos/df_general.csv', index=False)
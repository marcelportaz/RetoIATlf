import librosa
import numpy as np
import scipy.stats
import pandas as pd
import glob
import os

# ==========================================
# 1. FUNCIONES (Se definen fuera del bucle)
# ==========================================
def describe_freq(freqs):
    mean = np.mean(freqs)
    std = np.std(freqs) 
    maxv = np.amax(freqs) 
    minv = np.amin(freqs) 
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    # Extraemos el valor numérico exacto para evitar problemas con DataFrames
    mode = scipy.stats.mode(freqs, keepdims=True).mode[0]
    iqr = scipy.stats.iqr(freqs)
    
    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr]

def energy(x):
    return np.sum(x**2)

# ==========================================
# 2. CONFIGURACIÓN DE LA CARPETA
# ==========================================
ruta_carpeta = 'Audios/DS_LA_LAtrain/*.flac' 
archivos = glob.glob(ruta_carpeta)

# Aquí guardaremos todos los diccionarios
datos_features = [] 

print(f"Comenzando extracción de {len(archivos)} audios. Esto puede tardar un poco...")

# ==========================================
# 3. BUCLE DE EXTRACCIÓN
# ==========================================
for archivo in archivos:
    try:
        x, sr = librosa.load(archivo)
        freqs = np.fft.fftfreq(x.size)
        
        # --- Extracción de features (Tu código) ---
        stats_freq = describe_freq(freqs)
        rmse = librosa.feature.rms(y=x)[0]
        zero_crossings = sum(librosa.zero_crossings(y=x, pad=False))
        
        # Nota: Usamos feature.beat.tempo para evitar el FutureWarning
        tempo = librosa.beat.tempo(y=x, sr=sr)[0]
        
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
        
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        
        spec_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=x)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]

        # --- Empaquetado para Pandas ---
        # Asignamos cada valor a un nombre de columna
        features_audio = {
            'archivo': os.path.basename(archivo),
            'freq_mean': stats_freq[0],
            'freq_std': stats_freq[1],
            'freq_maxv': stats_freq[2],
            'freq_minv': stats_freq[3],
            'freq_median': stats_freq[4],
            'freq_skew': stats_freq[5],
            'freq_kurt': stats_freq[6],
            'freq_q1': stats_freq[7],
            'freq_q3': stats_freq[8],
            'freq_mode': stats_freq[9],
            'freq_iqr': stats_freq[10],
            'energy': energy(x),
            'rmse_mean': np.mean(rmse),
            'zero_crossings': zero_crossings,
            'tempo': float(tempo),
            'mfcc_mean': np.mean(mfcc),
            'tempogram_mean': np.mean(tempogram),
            'spec_centroid_mean': np.mean(spec_centroid),
            'spec_bandwidth_mean': np.mean(spectral_bandwidth),
            'spec_contrast_mean': np.mean(spectral_contrast),
            'spec_flatness_mean': np.mean(spectral_flatness),
            'spec_rolloff_mean': np.mean(spectral_rolloff)
        }
        
        datos_features.append(features_audio)
        
    except Exception as e:
        print(f"Error en {archivo}: {e}")

# ==========================================
# 4. CREACIÓN DEL DATAFRAME Y GUARDADO
# ==========================================
df_features = pd.DataFrame(datos_features)

# Guardar en CSV para no repetir este proceso
df_features.to_csv('mis_features_audio.csv', index=False)

print("\n¡Proceso terminado!")
print(f"Se ha creado un DataFrame con {df_features.shape[0]} filas (audios) y {df_features.shape[1]} columnas (features).")
import librosa
import numpy as np
import scipy.stats

x, sr = librosa.load('Audios/DS_LA_LAeval/LA_E_1041335.flac')

freqs = np.fft.fftfreq(x.size)

# Statistical features

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
    mode = scipy.stats.mode(freqs)[0]
    iqr = scipy.stats.iqr(freqs)
    
    return [mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr]

# Energy

def energy(x):
    return np.sum(x**2)

# Root Mean Square Energy

rmse = librosa.feature.rms(y=x)[0]

# Zero Crossing Rate

zero_crossings = sum(librosa.zero_crossings(y=x, pad=False))

# Tempo 

tempo = librosa.beat.tempo(y=x, sr=sr)[0]

# MFCCs

mfcc=librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)

# Tempogram
hop_length = 512
oenv = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

# Spectral centroid

spec_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]

# Spectral features

spectral_bandwidth=librosa.feature.spectral_bandwidth(y=x, sr=sr)[0]
spectral_contrast=librosa.feature.spectral_contrast(y=x, sr=sr)[0]
spectral_flatness=librosa.feature.spectral_flatness(y=x)[0]
spectral_rolloff=librosa.feature.spectral_rolloff(y=x, sr=sr)[0]

features = describe_freq(freqs)
features.append(energy(x))
features.append(np.mean(rmse))
features.append(zero_crossings)
features.append(tempo)
features.append(np.mean(mfcc))
features.append(np.mean(tempogram))
features.append(np.mean(spec_centroid))
features.append(np.mean(spectral_bandwidth))
features.append(np.mean(spectral_contrast))
features.append(np.mean(spectral_flatness))
features.append(np.mean(spectral_rolloff))
print(features)



"""#creame un plot donde poder ver la onda de audio y su espectro de frecuencias
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x)
plt.title('Waveform')
plt.subplot(2, 1, 2)
plt.plot(freqs, np.abs(np.fft.fft(x)))
plt.title('Frequency Spectrum')
plt.show()"""
# RetoIATlf
Reto Inteligencia Artificial Telefonica



## 1. Directory Structure
_______________________

  --> LA  
          --> ASVspoof2019_LA_asv_protocols
          --> ASVspoof2019_LA_asv_scores
	  --> ASVspoof2019_LA_cm_protocols
          --> ASVspoof2019_LA_dev
          --> ASVspoof2019_LA_eval
	  --> ASVspoof2019_LA_train
	  --> README.LA.txt


## 2. Description of the audio files
_________________________________

   ASVspoof2019_LA_train, ASVspoof2019_LA_dev, and ASVspoof2019_LA_eval contain audio files for training, development, and evaluation
   (LA_T_*.flac, LA_D_*.flac, and LA_E_*.flac, respectively). ASVspoof2019_PA_dev, and ASVspoof2019_PA_eval contain audio files to enroll ASV system. The audio files in the directories are in the flac format. 
   The sampling rate is 16 kHz, and stored in 16-bit.


## 3. Description of the protocols
_______________________________

CM protocols:

   ASVspoof2019_LA_cm_protocols contains protocol files in ASCII format for ASVspoof countermeasures:

   ASVspoof2019.LA.cm.train.trn.txt: training file list
   ASVspoof2019.LA.cm.dev.trl.txt: development trials
   ASVspoof2019.LA.cm.eval.trl.txt: evaluation trials 
	
   Each column of the protocol is formatted as:
   
   SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

   	1) SPEAKER_ID: 		LA_****, a 4-digit speaker ID
   	2) AUDIO_FILE_NAME: 	LA_****, name of the audio file
   	3) SYSTEM_ID: 		ID of the speech spoofing system (A01 - A19),  or, for bonafide speech SYSTEM-ID is left blank ('-')
   	4) -: 			This column is NOT used for LA.
	5) KEY: 		'bonafide' for genuine speech, or, 'spoof' for spoofing speech

   Note that: 
   
   	1) the third column is left blank (-) to make the structure coherent with physical access file list;
   	2) Brief description on LA spoofing systems, where TTS and VC denote text-to-speech and voice-conversion systems:
   	
        A01	TTS	neural waveform model
        A02	TTS	vocoder
        A03	TTS	vocoder
        A04	TTS	waveform concatenation
        A05	VC	vocoder
        A06	VC	spectral filtering
        		
        A07	TTS	vocoder+GAN
        A08	TTS	neural waveform
        A09	TTS	vocoder
        A10	TTS	neural waveform
        A11	TTS	griffin lim
        A12	TTS	neural waveform
        A13	TTS_VC	waveform concatenation+waveform filtering
        A14	TTS_VC	vocoder
        A15	TTS_VC	neural waveform
        A16	TTS	waveform concatenation
        A17	VC	waveform filtering
        A18	VC	vocoder
        A19	VC	spectral filtering
   
ASV protocols:
   
   ASVspoof2019_LA_asv_protocols contains the protocol files for ASV system

	ASVspoof2019.LA.asv.<1>.<2>.<3>.txt
	where
	<1> is either 'dev' or 'eval' based on whether the files describe the development or evaluation protocol,
	<2> ie either 'male (m)' or 'female (f)' separating the genders from each other or 'gender independent (gi)' 
	    contains trials for both genders (male trials followed by female trials),
	<3> is either 'trl' or 'trn' (trl = trial list, trn = speaker enrollment list).

	Trial (trl) file format for LA scenario:
	1st column: claimed speaker ID
	2nd column: test file ID
	3rd column: spoof attack ID (or 'bonafide' if the speech is not spoofed)
	4th column: key (target = target trial, nontarget = impostor trial, spoof = spoofing attack)

	Enrollment (trn) file format:
	1st column: ID of enrolled speaker
	2nd column: IDs of files used in the enrollment separated by commas
	

4. Baseline ASV scores
______________________

   ASVspoof2019_LA_asv_scores contains the scores calculated by a baseline ASV system for t-DCF evaluation
   
   	ASVspoof2019.LA.asv.dev.gi.trl.scores.txt:  scores given by the ASV system for development set data
   	ASVspoof2019.LA.asv.eval.gi.trl.scores.txt: scores given by the ASV system for evaluation set data
   	
   Each column is formatted as:
   
   CM_KEY ASV_KEY SCORES

   	1) CM_KEY: 		'bonafide' for genuine speech, or, the ID of the spoofing attack (A01 - A19)
   	2) ASV_KEY: 		'target' for claimed speaker, or, 'nontarget' for impostor speaker, or, 'spoof' for spoofing speech
   	3) SCORES: 		similarity score value

-------------------------------------------------------------------------------------------------------------------------------------

## 5. Audio Features Dictionary
_____________________________

This section provides a description of the metrics extracted from the audio files that are found in the CSV database (`mis_features_fusionadas.csv`).

- **energy**: The total energy of the audio signal.
- **rmse_mean**: Root Mean Square Energy. Measures the average energy of the audio frames.
- **zero_crossings**: The rate of sign-changes along a signal (how often the audio signal crosses the x-axis).
- **tempo**: Estimated tempo (beats per minute) of the audio.
- **mfcc_mean**: Mel-Frequency Cepstral Coefficients (average). Represents the short-term power spectrum of a sound.
- **tempogram_mean**: Average of the tempogram, which measures the local autocorrelation of the onset strength envelope.
- **spec_centroid_mean**: Spectral Centroid. Indicates where the "center of mass" of the spectrum is located (perceived brightness of a sound).
- **spec_bandwidth_mean**: Spectral Bandwidth. Describes the width of the frequency band of the audio signal.
- **spec_contrast_mean**: Spectral Contrast. Considers the spectral peak, spectral valley, and their difference in each frequency subband.
- **spec_flatness_mean**: Spectral Flatness. Measures how noise-like a sound is, as opposed to being tone-like.
- **spec_rolloff_mean**: Spectral Rolloff. The frequency below which a specified percentage of the total spectral energy lies.
- **freq_* (mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr)**: Descriptive statistical metrics calculated over the frequency distribution of the audio signal.

-------------------------------------------------------------------------------------------------------------------------------------

## 6. Machine Learning Models
___________________________

In our classification pipeline (`Redes_neuronales.py`), we implement and compare several distinct algorithms to identify whether an audio file is human (`bonafide`) or AI-generated (`spoof`), based on the extracted tabular features.

- **Random Forest**: An ensemble of decision trees trained in parallel. **Why we use it**: It serves as a highly robust baseline that is resistant to overfitting and works exceptionally well out-of-the-box with tabular audio features.
- **Gradient Boosting**: A sequential ensemble method that corrects errors from previous trees. **Why we use it**: It typically achieves higher predictive accuracy than Random Forest by building complex, fine-tuned decision boundaries, making it ideal for detecting subtle AI spoofing artifacts.
- **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane separating the classes. **Why we use it**: Highly effective in high-dimensional feature spaces. Using different kernels (like Linear or RBF) allows it to capture both simple and complex non-linear relationships in the audio metrics.
- **Multi-Layer Perceptron (MLP)**: A foundational Deep Learning approach (feedforward neural network). **Why we use it**: It learns deep, hierarchical representations from the scaled data. It is highly flexible and capable of modeling highly complex, latent patterns that traditional tree or distance-based algorithms might miss.

By comparing these models, we can determine the optimal balance between training speed, interpretability, and classification accuracy for audio deepfake detection.

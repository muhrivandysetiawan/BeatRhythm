class AudioFeatureExtractor:
    """
    AudioFeatureExtractor
    ---------------------
    Extracts and visualizes various audio features using Librosa.

    Supported features:
      - MFCC and Delta MFCC
      - Chroma
      - Spectral Flux
      - Mel-Spectrogram
      - RMSE (Energy)
      - Onset Strength

    Each feature can be visualized or directly returned for machine learning tasks.
    """

    def __init__(self, file_path, sr=22050):
        """Initialize audio file, load signal, and normalize amplitude."""
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.sr = sr
        self.y, _ = librosa.load(file_path, sr=self.sr)
        self.y = librosa.util.normalize(self.y)

    def plot_mfcc(self, n_mfcc=13, normalize=False, delta=False, delta_order=1,
                  cmap='magma', display=True):
        """Compute and plot MFCCs, with optional normalization and delta features."""
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)
        if normalize:
            mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / mfccs.std(axis=1, keepdims=True)

        if display:
            plt.figure(figsize=(15, 4))
            librosa.display.specshow(mfccs, sr=self.sr, x_axis='time', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'MFCC - {self.filename}')
            plt.tight_layout()
            plt.show()

        if delta and display:
            delta_mfccs = librosa.feature.delta(mfccs, order=delta_order)
            plt.figure(figsize=(15, 4))
            librosa.display.specshow(delta_mfccs, sr=self.sr, x_axis='time', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Delta MFCC (order={delta_order}) - {self.filename}')
            plt.tight_layout()
            plt.show()

        return mfccs

    def plot_chroma(self, hop_length=512, normalize=False, cmap='twilight', display=True):
        """Compute and plot the Chroma feature."""
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr, hop_length=hop_length)
        if normalize:
            chroma = (chroma - chroma.mean(axis=1, keepdims=True)) / chroma.std(axis=1, keepdims=True)

        if display:
            plt.figure(figsize=(15, 4))
            librosa.display.specshow(chroma, sr=self.sr, hop_length=hop_length,
                                     x_axis='time', y_axis='chroma', cmap=cmap)
            plt.colorbar(format='%+2.0f')
            plt.title(f'Chroma - {self.filename}')
            plt.tight_layout()
            plt.show()

        return chroma

    def plot_spectral_flux(self, hop_length=512, normalize=False, cmap='viridis', display=True):
        """Compute and plot the Spectral Flux along with the waveform."""
        D = np.abs(librosa.stft(self.y, hop_length=hop_length))
        D_norm = D / np.sum(D, axis=0, keepdims=True)
        flux = np.sqrt(np.sum((D_norm[:, 1:] - D_norm[:, :-1]) ** 2, axis=0))

        if normalize:
            flux = (flux - flux.mean()) / flux.std()

        if display:
            times_flux = librosa.frames_to_time(np.arange(len(flux)), sr=self.sr, hop_length=hop_length)
            times_wave = np.linspace(0, len(self.y) / self.sr, len(self.y))

            fig, ax1 = plt.subplots(figsize=(15, 3))
            ax1.plot(times_wave, self.y, color='blue', label='Waveform')
            ax1.set_ylabel('Amplitude', color='blue')
            ax1.set_xlabel('Time (s)')

            ax2 = ax1.twinx()
            ax2.plot(times_flux, flux, color='purple', label='Spectral Flux')
            ax2.set_ylabel('Spectral Flux', color='purple')

            ax1.set_title(f'Spectral Flux & Waveform - {self.filename}')
            plt.tight_layout()
            plt.show()

        return flux

    def plot_mel_spectrogram(self, n_mels=40, hop_length=256, n_fft=1024,
                             fmin=50, fmax=5000, display=True):
        """Compute and plot the Mel-Spectrogram in decibel scale."""
        mel_spec = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, fmin=fmin, fmax=fmax, window='blackman',
            center=False, pad_mode='constant'
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        if display:
            plt.figure(figsize=(20, 6))
            librosa.display.specshow(log_mel_spec, sr=self.sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Log-Mel Spectrogram - {self.filename}')
            plt.tight_layout()
            plt.show()

        return log_mel_spec

    def extract_rmse(self, hop_length=512, display=True):
        """Compute and optionally plot Root Mean Square Energy (RMSE)."""
        rmse = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]

        if display:
            times = librosa.frames_to_time(np.arange(len(rmse)), sr=self.sr, hop_length=hop_length)
            plt.figure(figsize=(12, 3))
            plt.plot(times, rmse, color='orange')
            plt.title(f'RMSE (Energy) - {self.filename}')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return rmse

    def extract_onset_strength(self, hop_length=512, display=True):
        """Compute and optionally plot Onset Strength Envelope."""
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop_length)

        if display:
            times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr, hop_length=hop_length)
            plt.figure(figsize=(12, 3))
            plt.plot(times, onset_env, color='red')
            plt.title(f'Onset Strength Envelope - {self.filename}')
            plt.xlabel('Time (s)')
            plt.ylabel('Onset Strength')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return onset_env

    def extract_all(self):
        """
        🚀 Extract all major features for ML-ready audio representation.
        Returns a dictionary containing mean values of each feature.
        """
        features = {
            "filename": self.filename,
            "mfcc_mean": self.plot_mfcc(display=True).mean(axis=1).tolist(),
            "chroma_mean": self.plot_chroma(display=True).mean(axis=1).tolist(),
            "flux_mean": float(self.plot_spectral_flux(display=True).mean()),
            "mel_spec_mean": float(self.plot_mel_spectrogram(display=True).mean()),
            "rmse_mean": float(self.extract_rmse(display=True).mean()),
            "onset_mean": float(self.extract_onset_strength(display=True).mean())
        }
        return features

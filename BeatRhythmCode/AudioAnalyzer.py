class AudioAnalyzer:
    """
    AudioAnalyzer
    --------------
    A class for advanced tempo, beat, and feature analysis using Madmom and Librosa.

    Capabilities:
      - Tempo estimation and histogram processing
      - Beat tracking using RNN and DBN
      - Signal, STFT, and spectrogram computation
      - Visualization of audio features and tempo treemap

    Parameters can be customized through dictionaries, or defaults are applied automatically.
    """

    def __init__(self, tempo_params=None, beat_params=None, signal_params=None,
                 stft_params=None, spectrogram_params=None):

        # Default parameters for the tempo histogram
        histogram_params = {
            'min_bpm': 40.0,
            'max_bpm': 250.0,
            'alpha': 0.79,
            'fps': 100,
            'transition_lambda': 100,
            'observation_lambda': 16,
            'online': False
        }
        self.histogram_processor = CombFilterTempoHistogramProcessor(**histogram_params)

        # Default parameters for tempo estimation
        tempo_default_params = {
            'act_smooth': 0.14,
            'hist_smooth': 9,
            'fps': 100,
            'interpolate': True,
            'histogram_processor': self.histogram_processor,
            'method': None
        }
        self.tempo_params = tempo_params or tempo_default_params

        # Default parameters for beat tracking
        beat_default_params = {
            'min_bpm': 55.0,
            'max_bpm': 215.0,
            'num_tempi': None,
            'transition_lambda': 100,
            'observation_lambda': 16,
            'threshold': 0,
            'correct': True,
            'fps': 100
        }
        self.beat_params = beat_params or beat_default_params

        # Default parameters for signal processing
        signal_default_params = {
            'sample_rate': 44100,
            'num_channels': 1,
            'norm': True,
            'gain': 0.0,
            'dtype': np.float32
        }
        self.signal_params = signal_params or signal_default_params

        # Default parameters for STFT
        stft_default_params = {
            'frame_size': 2048,
            'hop_size': 441,
            'window': np.hanning,
            'fps': 100,
            'circular_shift': True
        }
        self.stft_params = stft_params or stft_default_params

        # Default parameters for spectrogram
        spectrogram_default_params = {
            'num_bands': 24,
            'fmin': 20.0,
            'fmax': 20000.0,
            'norm_filters': True,
            'mul': 1.0,
            'add': 1.0,
            'fps': 100
        }
        self.spectrogram_params = spectrogram_params or spectrogram_default_params

        # Parameter validation
        self._validate_params()

        # Core processors
        self.signal_processor = SignalProcessor(**self.signal_params)
        self.stft_processor = ShortTimeFourierTransformProcessor(
            frame_size=self.stft_params['frame_size'],
            hop_size=self.stft_params['hop_size'],
            window=self.stft_params['window'],
            fps=self.stft_params['fps'],
            circular_shift=self.stft_params['circular_shift']
        )
        self.spectrogram_processor = LogarithmicFilteredSpectrogramProcessor(
            num_bands=self.spectrogram_params['num_bands'],
            fmin=self.spectrogram_params['fmin'],
            fmax=self.spectrogram_params['fmax'],
            norm_filters=self.spectrogram_params['norm_filters'],
            mul=self.spectrogram_params['mul'],
            add=self.spectrogram_params['add'],
            fps=self.spectrogram_params['fps']
        )

        # RNN + Tempo + Beat tracking processors
        self.rnn_processor = RNNBeatProcessor(fps=100, online=False, normalize=True)
        self.tempo_processor = TempoEstimationProcessor(**self.tempo_params)
        self.beat_tracker = DBNBeatTrackingProcessor(**self.beat_params)

    def _validate_params(self):
        """Validate user-defined or default parameters."""
        if self.stft_params['frame_size'] <= 0 or (
            self.stft_params['frame_size'] & (self.stft_params['frame_size'] - 1)
        ) != 0:
            raise ValueError("frame_size must be a positive power of 2")
        if self.stft_params['hop_size'] <= 0:
            raise ValueError("hop_size must be positive")
        if self.spectrogram_params['num_bands'] <= 0:
            raise ValueError("num_bands must be positive")
        if (self.spectrogram_params['fmin'] <= 0 or
                self.spectrogram_params['fmax'] <= self.spectrogram_params['fmin']):
            raise ValueError("fmin must be positive and fmax must be greater than fmin")

    def draw_bpm_treemap(self, tempi):
        """Draw a BPM treemap visualization showing tempo strength distribution."""
        bpm_values = [round(t[0]) for t in tempi]
        strengths = [t[1] for t in tempi]
        bpm_values, strengths = zip(*[(b, s) for b, s in zip(bpm_values, strengths) if s > 0])

        labels = [f"{round(bpm)}\n{strength:.1%}" for bpm, strength in zip(bpm_values, strengths)]
        colors = ['#2ca02c' if s >= 0.1 else '#ff7f0e' for s in strengths]

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)

        squarify.plot(
            sizes=strengths,
            label=labels,
            color=colors,
            alpha=0.85,
            text_kwargs={'fontsize': 12, 'weight': 'bold'},
            ax=ax,
            bar_kwargs={'linewidth': 2, 'edgecolor': 'white'}
        )

        plt.title("Tempo Treemap (BPM × Strength)", fontsize=14, pad=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def analyze(self, audio_file):
        """
        Perform a full analysis on a single audio file.

        Includes:
          - Signal processing
          - STFT
          - RNN activation
          - Tempo estimation
          - Beat tracking
          - Visualization
        """
        try:
            if not isinstance(audio_file, str):
                raise ValueError("audio_file must be a string path to an audio file.")
            if not os.path.isfile(audio_file):
                raise FileNotFoundError(f"Audio file '{audio_file}' not found.")

            print(f"Processing {audio_file}")

            signal = self.signal_processor(audio_file)
            stft = self.stft_processor(signal)
            activations = self.rnn_processor(signal)
            tempi = self.tempo_processor(activations)
            beats = self.beat_tracker(activations)

            print("Detected tempi (BPM, strength):")
            for tempo, strength in tempi:
                print(f"  - {tempo:.2f} BPM (strength: {strength:.3f})")

            print("\nBeat positions (seconds):")
            for i, beat in enumerate(beats[:5], 1):
                print(f"  - Beat {i}: {beat:.2f} seconds")
            if len(beats) > 5:
                print(f"  - ... (total {len(beats)} beats)")

            self.visualize(signal, stft, activations, beats, tempi)
            return tempi, beats, activations

        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def analyze_multiple(self, audio_files):
        """Run analysis for multiple audio files sequentially."""
        results = {}
        for audio_file in audio_files:
            print(f"\n=== Processing {audio_file} ===")
            tempi, beats, activations = self.analyze(audio_file)
            if tempi is not None and beats is not None:
                results[os.path.basename(audio_file)] = (tempi, beats, activations)

            time.sleep(3)
            gc.collect()
        return results

    def extract_features_only(self, audio_file):
        """
        🎯 Extract tempo and beat features without visualization.
        Ideal for batch or ML preprocessing.
        """
        try:
            if not isinstance(audio_file, str):
                raise ValueError("audio_file must be a string path.")
            if not os.path.isfile(audio_file):
                raise FileNotFoundError(f"File not found: {audio_file}")

            signal = self.signal_processor(audio_file)
            activations = self.rnn_processor(signal)
            tempi = self.tempo_processor(activations)
            beats = self.beat_tracker(activations)
            duration = len(signal) / signal.sample_rate

            return {
                'filename': os.path.basename(audio_file),
                'tempi': [(round(t[0], 2), round(t[1], 4)) for t in tempi],
                'bpm_main': round(tempi[0][0], 2) if tempi else None,
                'num_beats': len(beats),
                'duration': round(duration, 2),
                'beats': [round(b, 3) for b in beats]
            }

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None

    def visualize(self, signal, stft, activations, beats, tempi):
        """Visualize waveform, STFT, spectrogram, and beat activations."""
        try:
            spectrogram = self.spectrogram_processor(stft)
            stft_magnitude = np.abs(stft)
            y = signal
            sr = signal.sample_rate
            times = np.arange(len(activations)) / self.tempo_params['fps']

            fig, axs = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

            # 1. STFT Magnitude
            img1 = librosa.display.specshow(
                20 * np.log10(stft_magnitude.T + 1e-10),
                sr=sr,
                hop_length=int(self.stft_params['hop_size']),
                x_axis='time',
                y_axis='hz',
                cmap='magma',
                ax=axs[0]
            )
            axs[0].set_title('STFT Magnitude')
            fig.colorbar(img1, ax=axs[0], format='%+2.0f dB')

            # 2. Log-Filtered Spectrogram
            img2 = librosa.display.specshow(
                spectrogram.T,
                sr=sr,
                hop_length=int(self.stft_params['hop_size']),
                x_axis='time',
                y_axis='hz',
                cmap='magma',
                ax=axs[1]
            )
            axs[1].set_title('Logarithmic Filtered Spectrogram')
            fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

            # 3. Beat Activation Function
            axs[2].plot(times, activations, label='Beat Activations')
            axs[2].vlines(beats, ymin=0, ymax=1, colors='r', linestyles='--', label='Beats')
            axs[2].set_title('Beat Activation Function')
            axs[2].set_xlabel("Time (seconds)")
            axs[2].set_ylabel("Activation")

            # 4. Waveform and Beats
            axs[3].plot(np.linspace(0, len(y) / sr, len(y)), y, label='Waveform')
            axs[3].vlines(beats, ymin=-1, ymax=1, colors='r', linestyles='--', label='Beats')
            axs[3].set_title(f'Waveform and Beats (Primary Tempo: {tempi[0][0]:.2f} BPM)')
            axs[3].set_xlabel("Time (seconds)")
            axs[3].set_ylabel("Amplitude")

            plt.suptitle("Audio Feature Visualization", fontsize=16, y=1.02)
            plt.show()

            # Draw BPM treemap
            time.sleep(0.5)
            self.draw_bpm_treemap(tempi)

        except Exception as e:
            print(f"Visualization failed: {e}")

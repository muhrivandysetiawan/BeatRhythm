class LibrosaAudioProcessor:
    """
    LibrosaAudioProcessor
    ---------------------
    Kelas ini memproses kumpulan file audio (WAV, MP3, dsb) menggunakan Librosa.
    Setiap file dianalisis untuk menghasilkan fitur dasar seperti durasi, centroid,
    bandwidth, dan amplitudo rata-rata. Hasil dapat disimpan ke cache atau diekspor ke JSON.

    Fitur:
      - Multi-threaded audio processing
      - Caching hasil ekstraksi fitur
      - Ekspor ringkasan hasil ke JSON
    """

    def __init__(self, sr=44100, cache_dir="audio_cache", use_cache=True, max_workers=4):
        """Inisialisasi parameter dasar dan direktori cache."""
        self.sample_rate = sr
        self.data = {}
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_workers = max_workers

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, filename):
        """Buat path file cache (.pkl) dari nama file audio."""
        base = os.path.splitext(os.path.basename(filename))[0]
        return os.path.join(self.cache_dir, f"{base}.pkl")

    def _process_single_file(self, file_path):
        """Proses satu file audio: load, ekstraksi fitur, dan simpan ke cache."""
        cache_path = self._cache_path(file_path)

        # Gunakan cache jika tersedia
        if self.use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return os.path.basename(file_path), pickle.load(f)
            except Exception as e:
                print(f"⚠️ Cache corrupt for {file_path} — Reloading. {e}")

        # Load dan proses ulang file audio
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            y = librosa.util.normalize(y)

            duration = librosa.get_duration(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            amplitude = np.mean(np.abs(y))

            result = {
                'signal': y,
                'sample_rate': sr,
                'duration': float(duration),
                'centroid': float(spectral_centroid),
                'bandwidth': float(spectral_bandwidth),
                'amplitude': float(amplitude)
            }

            # Simpan ke cache
            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)

            return os.path.basename(file_path), result

        except Exception as e:
            print(f"❌ Gagal proses {file_path} — {e}")
            return os.path.basename(file_path), None

    def process_files(self, file_list):
        """Proses beberapa file audio secara paralel menggunakan ThreadPoolExecutor."""
        print(f"\n🎶 Memproses {len(file_list)} file audio dengan {self.max_workers} thread...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._process_single_file, file_list)

        for filename, result in results:
            if result:
                self.data[filename] = result
                dur = result['duration']
                print(f"✅ {filename}")
                print(f"   🕒 Durasi    : {int(dur // 60)}m {int(dur % 60)}s ({dur:.2f} s)")
                print(f"   📊 Amplitudo : {result['amplitude']:.4f}")
                print(f"   🎯 Centroid  : {result['centroid']:.2f} Hz")
                print(f"   🎯 Bandwidth : {result['bandwidth']:.2f} Hz\n")
            else:
                print(f"🚫 Gagal proses: {filename}")

        return self.data

    def summary(self):
        """Tampilkan ringkasan file audio yang sudah berhasil diproses."""
        print("\n📦 Data Tersimpan:")
        for filename, info in self.data.items():
            print(f"🎵 {filename} — {len(info['signal'])} samples @ {info['sample_rate']} Hz")

    def get_dataset(self):
        """Kembalikan dataset hasil pemrosesan audio."""
        return self.data

    def export_to_json(self, json_path="processed_audio_summary.json"):
        """Ekspor ringkasan fitur audio ke file JSON."""
        summary = {
            fname: {
                "duration": float(round(info['duration'], 2)),
                "centroid": float(round(info['centroid'], 2)),
                "bandwidth": float(round(info['bandwidth'], 2)),
                "amplitude": float(round(info['amplitude'], 4))
            }
            for fname, info in self.data.items()
        }

        # Penulisan file dengan error handling
        try:
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"✅ Summary JSON disimpan ke: {json_path}")
        except IOError as e:
            print(f"❌ Error writing JSON to {json_path}: {e}")

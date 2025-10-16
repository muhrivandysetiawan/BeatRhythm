class BeatRhyhtmDataLoader:
    """
    BeatRhythmDataLoader
    --------------------
    Loader untuk mencocokkan file WAV dan MIDI dari dua folder terpisah, 
    serta menyimpannya ke dalam cache JSON agar proses scanning lebih cepat.
    
    Fitur:
      - Otomatis mount Google Drive (jika belum)
      - Scan folder WAV dan MIDI
      - Mencocokkan file dengan nama sama
      - Menyimpan hasil ke cache JSON
      - Menyediakan fungsi validasi dan ringkasan data
    """

    def __init__(self, root_dir="/content/drive/MyDrive/BeatSaberAudio",
                 wav_folder="WAVN", midi_folder="MIDIN", verbose=True):
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, wav_folder)
        self.midi_dir = os.path.join(root_dir, midi_folder)
        self.verbose = verbose

        self.audio_files = []
        self.midi_files = []
        self.song_names = []
        self.pairs = []

        self._mount_drive()
        self._load_and_match_files()

    def _mount_drive(self):
        """Mount Google Drive jika belum terpasang."""
        if not os.path.exists("/content/drive"):
            if self.verbose:
                print("🔗 Mounting Google Drive...")
            drive.mount("/content/drive")
        else:
            if self.verbose:
                print("✅ Google Drive already mounted.")

    def _load_and_match_files(self):
        """Scan folder WAV & MIDI, cocokan nama file, lalu simpan hasil ke cache."""
        cache_path = os.path.join(self.root_dir, "pairs_cache.json")

        # Gunakan cache jika tersedia
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache_data = json.load(f)
                self.pairs = cache_data["pairs"]
                self.song_names = cache_data["song_names"]
                self.audio_files = cache_data["audio_files"]
                self.midi_files = cache_data["midi_files"]
            if self.verbose:
                print(f"📦 Loaded cached data from {cache_path}")
            return

        # Scan folder jika cache belum ada
        if self.verbose:
            print(f"\n📂 Scanning folders:\n- WAV : {self.wav_dir}\n- MIDI: {self.midi_dir}")

        wavs = sorted([
            f for f in os.listdir(self.wav_dir) if f.lower().endswith('.wav')
        ])
        midis = sorted([
            f for f in os.listdir(self.midi_dir) if f.lower().endswith(('.midi', '.mid'))
        ])

        midi_dict = {
            os.path.splitext(f)[0]: os.path.join(self.midi_dir, f) for f in midis
        }

        # Cocokkan nama WAV dan MIDI
        for wav in wavs:
            base = os.path.splitext(wav)[0]
            if base in midi_dict:
                wav_path = os.path.join(self.wav_dir, wav)
                midi_path = midi_dict[base]
                self.audio_files.append(wav_path)
                self.midi_files.append(midi_path)
                self.song_names.append(base)
                self.pairs.append((wav_path, midi_path))
            else:
                if self.verbose:
                    print(f"⚠️ MIDI not found for: {base}")

        # Tampilkan hasil pencocokan
        if self.verbose:
            print(f"\n✅ Matched {len(self.pairs)} WAV+MIDI pairs.")
            for i, name in enumerate(self.song_names):
                print(f"{i+1:>2}. 🎶 {name}")

        # Simpan hasil ke cache
        cache_data = {
            "pairs": self.pairs,
            "song_names": self.song_names,
            "audio_files": self.audio_files,
            "midi_files": self.midi_files
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        if self.verbose:
            print(f"💾 Cache saved to {cache_path}")

    def get_pairs(self):
        """Kembalikan daftar pasangan file (WAV, MIDI)."""
        return self.pairs

    def validate_files(self):
        """Validasi keberadaan semua file WAV dan MIDI."""
        if not self.verbose:
            return
        print("\n🔍 Validasi File WAV & MIDI:")
        for audio_path, midi_path in zip(self.audio_files, self.midi_files):
            filename = os.path.basename(audio_path)
            wav_status = "✅ WAV" if os.path.exists(audio_path) else "❌ WAV"
            midi_status = "✅ MIDI" if os.path.exists(midi_path) else "❌ MIDI"
            print(f"🎵 {filename:<30} | {wav_status} | {midi_status}")

    def summary(self):
        """Tampilkan ringkasan jumlah data dan lokasi folder."""
        if not self.verbose:
            return
        print("\n📊 Data Summary:")
        print(f"- Total Songs  : {len(self.pairs)}")
        print(f"- WAV Folder   : {self.wav_dir}")
        print(f"- MIDI Folder  : {self.midi_dir}")

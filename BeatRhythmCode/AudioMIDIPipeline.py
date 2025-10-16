class AudioMIDIPipeline:
    """
    AudioMIDIPipeline
    -----------------
    A unified pipeline to process, analyze, and compare multiple audio (WAV/MP3)
    and MIDI files. It performs tempo and beat analysis, compares the detected beats
    with MIDI notes, generates structured block JSONs, and exports both logs and reports.

    Features:
      - Automatic audio–MIDI matching
      - Beat-based precision evaluation
      - Block JSON generation per song
      - Combined report and logging output
    """

    def __init__(self, audio_files, midi_files, tolerance=0.1, verbose=True,
                 log_file="log_pipeline.txt", block_dir="blocks", merged_file="all_blocks.json"):
        self.audio_files = audio_files
        self.midi_files = midi_files
        self.tolerance = tolerance
        self.verbose = verbose
        self.log_file = log_file
        self.block_dir = block_dir
        self.merged_file = merged_file

        self.analyzer = AudioAnalyzer()
        self.results = {}
        self.log_messages = []
        self.all_blocks = []

        os.makedirs(self.block_dir, exist_ok=True)

    def log(self, msg):
        """Log messages to console and internal buffer."""
        self.log_messages.append(msg)
        if self.verbose:
            print(msg)

    def _grade_precision(self, precision):
        """Assign a letter grade based on precision percentage."""
        if precision >= 90: return "A"
        elif precision >= 75: return "B"
        elif precision >= 60: return "C"
        elif precision >= 45: return "D"
        else: return "E"

    def _find_midi_match(self, wav_name):
        """Find a matching MIDI file based on the audio filename."""
        base_name = os.path.basename(wav_name).split('.')[0]
        for midi in self.midi_files:
            if base_name in os.path.basename(midi):
                return midi
        return None

    def run(self):
        """
        Run the entire pipeline:
          1. Analyze audio files
          2. Compare with corresponding MIDI files
          3. Generate and export block JSON files
          4. Save merged results and process logs
        """
        self.results = self.analyzer.analyze_multiple(self.audio_files)

        for wav_name, (tempi, beats, _) in self.results.items():
            match_midi = self._find_midi_match(wav_name)
            if match_midi is None:
                self.log(f"⚠️ No matching MIDI for {wav_name}")
                continue

            try:
                # Step 1: Compare audio beats with MIDI notes
                comparator = BeatMIDIComparator(match_midi, beats, tolerance=self.tolerance)
                total_notes = len(comparator.notes)
                precision = (len(comparator.matches) / total_notes * 100) if total_notes > 0 else 0.0
                grade = self._grade_precision(precision)

                self.log(f"\n🎧 File: {wav_name}")
                self.log(f"🎯 Precision: {precision:.2f}% | Grade: {grade}")
                comparator.summary()

                # Step 2: Generate block JSON file
                gen = BlockGeneratorHybrid(
                    midi_path=match_midi,
                    audio_path=wav_name,
                    beats=beats,
                    tempo=tempi[0][0],
                    alignment_map=comparator.matches,
                    strict_alignment=False
                )
                gen.generate()
                gen.summary()

                # Save per-song block file
                name = os.path.splitext(os.path.basename(wav_name))[0].replace(" ", "_")
                song_path = os.path.join(self.block_dir, f"{name}_blocks.json")
                json_data = gen.export(song_path)

                if json_data:
                    self.all_blocks.append(json_data)

                gen.cleanup()
                del gen, comparator, tempi, beats
                gc.collect()

            except Exception as e:
                self.log(f"❌ Error processing {wav_name}: {e}")

        # Step 3: Save combined results
        with open(self.merged_file, "w") as f:
            json.dump(self.all_blocks, f, indent=2)

        self.log(f"✅ All blocks saved in '{self.block_dir}/'")
        self.log(f"✅ Merged file saved as '{self.merged_file}'")

        # Step 4: Save logs
        with open(self.log_file, "w") as f:
            f.write("\n".join(self.log_messages))

    def export_report(self, filename="comparison_results.csv"):
        """
        Export a precision and grade report (Audio–MIDI comparison) to CSV.
        """
        rows = []
        for wav_name, (tempi, beats, _) in self.results.items():
            midi = self._find_midi_match(wav_name)
            if midi is None:
                continue

            comp = BeatMIDIComparator(midi, beats, self.tolerance)
            matched = len(comp.matches)
            total = len(comp.notes)
            precision = (matched / total * 100) if total > 0 else 0.0
            grade = self._grade_precision(precision)

            rows.append({
                'File': wav_name,
                'Tempo': tempi[0][0],
                'Beats': len(beats),
                'Notes': total,
                'Matches': matched,
                'Precision (%)': round(precision, 2),
                'Grade': grade
            })

        pd.DataFrame(rows).to_csv(filename, index=False)
        self.log(f"✅ Report saved to {filename}")

class BeatMIDIComparator:
    """
    BeatMIDIComparator
    -------------------
    A professional comparator class for evaluating alignment between extracted WAV beats 
    and note onset times from a MIDI file.

    Features:
      - Adaptive tolerance system based on local tempo
      - Matching and mismatch analysis for beats vs. MIDI notes
      - Precision calculation and summary reporting
      - Multiple visualization tools (timeline, histogram, scatter, heatmap, alignment lines)
      - Optional summary export to CSV
    """

    def __init__(self, midi_path, beats, tolerance=0.1, adaptive_tolerance=True, save_path=None):
        self.midi_path = midi_path
        self.beats = np.array(beats)
        self.tolerance = tolerance
        self.adaptive_tolerance = adaptive_tolerance
        self.save_path = save_path

        self.notes = self._load_midi_notes()
        self.matches = []
        self.mismatches = []
        self.unused_beats = []

        self._compare()

    # --------------------------------------------------------------
    # MIDI Loading & Processing
    # --------------------------------------------------------------

    def _load_midi_notes(self):
        """Load note onset times from the given MIDI file using PrettyMIDI."""
        try:
            midi = pretty_midi.PrettyMIDI(self.midi_path)
            note_onsets = sorted([note.start for inst in midi.instruments for note in inst.notes])
            return np.array(note_onsets)
        except Exception as e:
            print(f"❌ Failed to load MIDI: {e}")
            return np.array([])

    # --------------------------------------------------------------
    # Adaptive Tolerance Function
    # --------------------------------------------------------------

    def _adaptive_tolerance_fn(self, note_time):
        """Compute adaptive tolerance based on local BPM."""
        if len(self.beats) < 2:
            return self.tolerance

        beat_diffs = np.diff(self.beats)
        local_bpm = 60.0 / np.median(beat_diffs)
        adapt_tol = self.tolerance * (60 / local_bpm)

        # Clamp tolerance to 10–250ms
        return max(0.01, min(float(adapt_tol), 0.25))

    # --------------------------------------------------------------
    # Comparison Logic
    # --------------------------------------------------------------

    def _compare(self):
        """Perform beat-note matching and store matched/mismatched data."""
        self.matches = []
        self.mismatches = []
        used_beat_indices = set()

        for note_time in self.notes:
            diffs = np.abs(self.beats - note_time)
            sorted_idxs = np.argsort(diffs)
            matched = False

            # Use adaptive or fixed tolerance
            tolerance_used = (
                self._adaptive_tolerance_fn(note_time)
                if self.adaptive_tolerance
                else self.tolerance
            )

            for idx in sorted_idxs:
                if idx in used_beat_indices:
                    continue
                if diffs[idx] <= tolerance_used:
                    self.matches.append((note_time, self.beats[idx]))
                    used_beat_indices.add(idx)
                    matched = True
                    break

            if not matched:
                closest_idx = sorted_idxs[0]
                self.mismatches.append((note_time, self.beats[closest_idx]))

        all_indices = set(range(len(self.beats)))
        self.unused_beats = list(all_indices - used_beat_indices)

    # --------------------------------------------------------------
    # Summary & CSV Export
    # --------------------------------------------------------------

    def _save_summary_to_csv(self, precision):
        """Save alignment summary as a CSV file if a save path is provided."""
        if not self.save_path:
            print("⚠️ Save path not specified. Skipping CSV summary save.")
            return

        try:
            csv_file = os.path.join(
                self.save_path, f"{os.path.basename(self.midi_path)}_summary.csv"
            )

            rows = [{
                "File": os.path.basename(self.midi_path),
                "Total Notes": len(self.notes),
                "Matched Beats": len(self.matches),
                "Unmatched Notes": len(self.mismatches),
                "Unused Beats": len(self.unused_beats),
                "Precision (%)": round(precision, 2)
            }]

            df = pd.DataFrame(rows)
            os.makedirs(self.save_path, exist_ok=True)
            df.to_csv(csv_file, index=False)
            print(f"✅ Summary saved to {csv_file}")

        except Exception as e:
            print(f"❌ Failed to save summary: {e}")

    def summary(self):
        """Print precision metrics and offset statistics."""
        total = len(self.notes)
        matched = len(self.matches)
        unmatched = len(self.mismatches)
        precision = (matched / total * 100) if total > 0 else 0

        print(f"📊 Beat-MIDI Alignment Summary for {os.path.basename(self.midi_path)}")
        print(f"🎵 Total Notes       : {total}")
        print(f"✅ Matched Beats     : {matched}")
        print(f"❌ Unmatched Notes   : {unmatched}")
        print(f"🟥 Unused Beats      : {len(self.unused_beats)}")
        print(f"🎯 Precision         : {precision:.2f}%")

        if matched > 0:
            diffs = [abs(n - b) for n, b in self.matches]
            print(f"⏱️  Avg Offset       : {np.mean(diffs)*1000:.2f} ms (std: {np.std(diffs)*1000:.2f} ms)")

        if self.save_path:
            self._save_summary_to_csv(precision)

    # --------------------------------------------------------------
    # Visualization Functions
    # --------------------------------------------------------------

    def plot_timeline(self, show_window=15):
        """Plot note and beat positions on a common timeline."""
        plt.figure(figsize=(14, 4))
        plt.vlines(self.beats, ymin=0, ymax=1, colors='red', alpha=0.6, label='WAV Beats')
        plt.vlines(self.notes, ymin=0, ymax=0.6, colors='blue', alpha=0.7, label='MIDI Notes')
        plt.hlines(y=0.1, xmin=self.notes.min()-0.1, xmax=self.notes.max()+0.1, colors='black', linestyles='dotted')
        plt.xlim(0, show_window)
        plt.title(f'MIDI vs WAV Beats ({os.path.basename(self.midi_path)})')
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def plot_offset_histogram(self):
        """Show histogram of time offsets between matched MIDI notes and WAV beats."""
        if not self.matches:
            print("⚠️ No matches found.")
            return

        offsets_ms = [abs(n - b) * 1000 for n, b in self.matches]
        plt.figure(figsize=(8, 3))
        sns.histplot(offsets_ms, bins='auto', kde=True, color="purple")
        plt.axvline(self.tolerance * 1000, color='red', linestyle='--',
                    label=f'Tolerance = {self.tolerance*1000:.0f}ms')
        plt.title("Offset Histogram (Matched Notes)")
        plt.xlabel("Offset (ms)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_offset_scatter(self):
        """Scatter plot of offsets over time for visualizing drift or misalignment."""
        if not self.matches:
            print("⚠️ No matches found.")
            return

        times = [n for n, _ in self.matches]
        offsets = [(n - b) * 1000 for n, b in self.matches]

        plt.figure(figsize=(12, 3))
        plt.axhline(0, color='gray', linestyle='--')
        plt.axhline(self.tolerance * 1000, color='red', linestyle='--', linewidth=0.8)
        plt.axhline(-self.tolerance * 1000, color='red', linestyle='--', linewidth=0.8)
        sns.scatterplot(x=times, y=offsets, color="teal", s=20)
        plt.title("Offset Scatter: MIDI vs WAV Beat (per Time)")
        plt.xlabel("Time (s)")
        plt.ylabel("Offset (ms)")
        plt.tight_layout()
        plt.show()

    def plot_offset_heatmap(self, bucket_size=2.0, max_time=None):
        """Display offset patterns as a heatmap across defined time buckets."""
        if not self.matches:
            print("⚠️ No matches found.")
            return

        times = np.array([n for n, _ in self.matches])
        offsets = np.array([(n - b) * 1000 for n, b in self.matches])

        if max_time is None:
            max_time = times.max()

        buckets = np.arange(0, max_time + bucket_size, bucket_size)
        df = pd.DataFrame({'time': times, 'offset': offsets})
        df['bucket'] = pd.cut(df['time'].tolist(), bins=buckets.tolist())

        heat_data = df.groupby('bucket', observed=True)['offset'].apply(list).apply(pd.Series).T
        plt.figure(figsize=(12, 3))
        sns.heatmap(heat_data, cmap="coolwarm", cbar_kws={'label': 'Offset (ms)'})
        plt.title("Offset Heatmap per Time Window")
        plt.xlabel("Time Bucket")
        plt.ylabel("Sample Index")
        plt.tight_layout()
        plt.show()

    def plot_alignment_lines(self, window=15):
        """Visualize beat-note matching relationships with connecting lines."""
        plt.figure(figsize=(12, 4))

        for note, beat in self.matches:
            if note <= window:
                plt.plot([note, beat], [0.3, 0.7], color='green', alpha=0.4)

        for note, beat in self.mismatches:
            if note <= window:
                plt.plot([note, beat], [0.3, 0.7], color='red', alpha=0.5, linestyle='--')

        plt.vlines(self.notes[self.notes <= window], ymin=0, ymax=0.3,
                   color='blue', alpha=0.6, label='MIDI Notes')
        plt.vlines(self.beats[self.beats <= window], ymin=0.7, ymax=1.0,
                   color='red', alpha=0.6, label='WAV Beats')

        plt.ylim(0, 1.1)
        plt.title(f'MIDI-WAV Beat Alignment Lines (≤ {window}s)')
        plt.xlabel('Time (s)')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def plot_all(self, window=15):
        """Render all visualization methods for a comprehensive analysis view."""
        self.plot_timeline(show_window=window)
        self.plot_alignment_lines(window=window)
        self.plot_offset_scatter()
        self.plot_offset_histogram()
        self.plot_offset_heatmap(max_time=window)

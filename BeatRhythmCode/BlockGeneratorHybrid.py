class BlockGeneratorHybrid:
    """
    BlockGeneratorHybrid
    -------------------
    A class for generating beat-synchronized blocks (notes, bombs, walls) from MIDI and audio files for rhythm-based applications.
    It processes MIDI notes, aligns them with audio beats, and generates structured block data with dynamic direction and color assignments.
    The class supports visualization, difficulty estimation, and JSON export of generated blocks.

    Features:
      - MIDI note parsing and beat alignment
      - Dynamic block generation (notes, bombs, walls) with probabilistic direction and color variation
      - Prevention of repetitive patterns and enforcement of minimum time gaps
      - Visualization of block distributions and heatmaps
      - JSON export and summary reporting
    """

    def __init__(self, midi_path, audio_path, beats, tempo, pitch_threshold=60, time_interval=0.5, fps=100, alignment_map=None, strict_alignment=False):
        self.midi_path = midi_path
        self.audio_path = audio_path
        self.filename = os.path.basename(audio_path)
        self.pitch_threshold = pitch_threshold
        self.time_interval = time_interval
        self.fps = fps
        self.blocks = []
        self.beats = np.array(beats)
        self.tempo = tempo
        self.activations = None
        self._load_midi()
        self.alignment_map = alignment_map if alignment_map else []
        self.strict_alignment = strict_alignment
        self.MIN_GAP = 0.101
        self.TIME_DECIMALS = 3
        self.min_gap = self.MIN_GAP
        self.time_decimals = self.TIME_DECIMALS

    def _load_midi(self):
        """Load and parse MIDI file, extracting notes with start time, pitch, and velocity."""
        try:
            self.midi = pretty_midi.PrettyMIDI(self.midi_path)
            self.notes = []
            for inst in self.midi.instruments:
                for note in inst.notes:
                    self.notes.append((note.start, note.pitch, note.velocity))
            self.notes.sort()
            print(f"✅ Loaded {len(self.notes)} notes from {os.path.basename(self.midi_path)}")
        except Exception as e:
            print(f"❌ Failed to load MIDI {os.path.basename(self.midi_path)}: {e}")
            self.notes = []

    def get_direction_by_pitch(self, pitch):
        """Assign a block direction based on MIDI note pitch with probabilistic variation."""
        if pitch < 48:
            base_dir = 0
        elif pitch < 60:
            base_dir = 3
        elif pitch < 72:
            base_dir = 2
        elif pitch < 84:
            base_dir = 1
        else:
            base_dir = random.choice([4,5,6,7])
        if random.random() < 0.3:
            return random.choice([d for d in range(9) if d != base_dir])
        return base_dir

    def is_valid_direction_transition(self, last_dir, new_dir):
        """Validate direction transitions to prevent sharp or invalid changes."""
        if last_dir is None:
            return True
        invalid_transitions = {
            (0, 1), (1, 0),
            (2, 3), (3, 2),
            (4, 7), (7, 4),
            (5, 6), (6, 5),
        }
        return (last_dir, new_dir) not in invalid_transitions

    def _generate_notes(self):
        """Generate note blocks with dynamic color and direction assignments."""
        last_time = {0: -self.time_interval, 1: -self.time_interval}
        last_block = {0: {}, 1: {}}
        last_hand = 1
        last_pitch = None
        last_dir = {0: None, 1: None}
        for start_time, pitch, velocity in self.notes:
            red_count = sum(1 for b in self.blocks if b.get("color") == 0)
            blue_count = sum(1 for b in self.blocks if b.get("color") == 1)
            total_blocks = red_count + blue_count
            red_ratio = red_count / total_blocks if total_blocks > 0 else 0.5
            blue_ratio = blue_count / total_blocks if total_blocks > 0 else 0.5
            if red_ratio > 0.6:
                hand = 1
            elif blue_ratio > 0.6:
                hand = 0
            else:
                if last_pitch is None or abs(pitch - last_pitch) >= 3:
                    hand = 1 - last_hand
                else:
                    hand = 1 - last_hand if (start_time - last_time[last_hand]) > 0.08 else last_hand
            if len(self.blocks) >= 2:
                last_colors = [b["color"] for b in self.blocks[-2:] if b["type"] == "note"]
                if last_colors and last_colors[0] == last_colors[1] == hand:
                    hand = 1 - hand
            dir_counter = Counter([b['direction'] for b in self.blocks if b['type'] == 'note'])
            most_common_dir, freq = dir_counter.most_common(1)[0] if dir_counter else (None, 0)
            if freq > 0.4 * len(self.blocks):
                direction = random.choice([d for d in range(9) if d != most_common_dir])
            else:
                candidate_dir = self.get_direction_by_pitch(pitch)
                if not self.is_valid_direction_transition(last_dir[hand], candidate_dir):
                    direction = 8
                else:
                    direction = candidate_dir
            block = {
                "time": start_time,
                "type": "note",
                "color": hand,
                "direction": direction
            }
            beat_duration = 60 / self.tempo
            dynamic_interval = max(
                beat_duration * (0.3 if self.tempo > 120 else 0.5),
                self.MIN_GAP
            )
            if start_time - last_time[hand] >= dynamic_interval:
                jitter = 0.0
                if last_block[hand] and abs(start_time - last_block[hand].get("time", 0.0)) < 0.01:
                    jitter = round(random.uniform(-0.005, 0.005), self.TIME_DECIMALS)
                proposed_time = start_time + jitter
                if last_time[hand] is not None and round(proposed_time - last_time[hand], self.TIME_DECIMALS) < self.MIN_GAP:
                    proposed_time = round(last_time[hand] + self.MIN_GAP, self.TIME_DECIMALS)
                else:
                    proposed_time = round(proposed_time, self.TIME_DECIMALS)
                block["time"] = proposed_time
                self.blocks.append(block)
                last_block[hand] = block
                last_time[hand] = block["time"]
                last_dir[hand] = direction
            last_pitch = pitch
            last_hand = hand

    def _generate_bombs(self, velocity_threshold=100):
        """Generate bomb blocks based on note pitch and velocity criteria."""
        bomb_candidates = []
        for start_time, pitch, velocity in self.notes:
            if pitch < 40 or pitch > 96 or velocity > velocity_threshold:
                position = random.choice(["left", "center", "right"])
                bomb_candidates.append({
                    "time": start_time,
                    "type": "bomb",
                    "position": position
                })
        max_bombs = int(0.3 * len(self.notes))
        selected_bombs = bomb_candidates[:max_bombs]
        self.blocks.extend(selected_bombs)

    def _generate_walls(self, wall_threshold=0.8, wall_duration=0.6):
        """Generate wall blocks in gaps between beats without notes."""
        wall_count = 0
        for i in range(1, len(self.beats)):
            current_beat = self.beats[i]
            last_beat = self.beats[i-1]
            gap = current_beat - last_beat
            if gap >= wall_threshold:
                has_note_between = any(
                    last_beat <= note_time <= current_beat
                    for note_time, _, _ in self.notes
                )
                if not has_note_between:
                    wall_time = last_beat
                    wall_position = random.choice(["left", "center", "right"])
                    self.blocks.append({
                        "time": wall_time,
                        "type": "wall",
                        "duration": min(gap - 0.2, wall_duration),
                        "position": wall_position
                    })
                    wall_count += 1
        min_walls = 5 if self.notes[-1][0] > 60 else 3
        while wall_count < min_walls:
            wall_time = self.beats[wall_count * len(self.beats) // min_walls]
            self.blocks.append({
                "time": wall_time,
                "type": "wall",
                "duration": 0.5,
                "position": random.choice(["left", "center", "right"])
            })
            wall_count += 1

    def _group_blocks_by_time(self, tolerance=0.01):
        """Group blocks by time proximity for alignment and spacing checks."""
        if not self.blocks:
            return []
        sorted_blocks = sorted(self.blocks, key=lambda x: x['time'])
        groups = []
        current_group = [sorted_blocks[0]]
        for block in sorted_blocks[1:]:
            if abs(block['time'] - current_group[-1]['time']) <= tolerance:
                current_group.append(block)
            else:
                groups.append(current_group)
                current_group = [block]
        groups.append(current_group)
        return groups

    def _align_blocks_to_beats(self, max_offset=0.3, max_blocks_per_time=4):
        """
        Align blocks to the nearest beat while enforcing minimum time gaps and limiting blocks per time slot.
        Handles edge cases like missing beats or high note density.
        """
        aligned = []
        if getattr(self, "beats", None) is None or self.beats.size == 0:
            print(f"⚠️ Skipping beat alignment: No beats available for {self.filename}.")
            return
        MIN_GAP = getattr(self, "MIN_GAP", 0.101)
        TIME_DECIMALS = getattr(self, "TIME_DECIMALS", 3)
        beat_gaps = np.diff(self.beats)
        median_gap = float(np.median(beat_gaps)) if len(beat_gaps) > 0 else 0.5
        note_duration_span = (self.notes[-1][0] - self.notes[0][0]) if len(self.notes) > 1 else 1.0
        note_density = len(self.notes) / note_duration_span if note_duration_span > 0 else 1.0
        dynamic_offset = min(max_offset, float(median_gap * (0.5 + 0.2 * note_density)))
        print(f"🔧 Using dynamic offset: {dynamic_offset:.3f}s for beat alignment (max: {max_offset}s)")
        initially_aligned_blocks = []
        for block in self.blocks:
            t = float(block.get("time", 0.0))
            closest_beat_index = int(np.argmin(np.abs(self.beats - t)))
            closest_beat_time = float(self.beats[closest_beat_index])
            offset = abs(closest_beat_time - t)
            block["original_time"] = block.get("time", t)
            block["offset"] = round(offset, TIME_DECIMALS)
            should_snap = (offset <= dynamic_offset and block.get("type") == "note") or \
                          (self.strict_alignment and offset <= dynamic_offset)
            if should_snap:
                beat_target = round(closest_beat_time, TIME_DECIMALS)
                conflict = any(abs(b.get("time", 0.0) - beat_target) < MIN_GAP for b in initially_aligned_blocks)
                if not conflict:
                    block["time"] = beat_target
                else:
                    block["time"] = round(beat_target + MIN_GAP, TIME_DECIMALS)
            else:
                block["time"] = round(t, TIME_DECIMALS)
            initially_aligned_blocks.append(block)
        grouped_blocks = self._group_blocks_by_time(tolerance=MIN_GAP * 0.5)
        final_blocks = []
        for group in grouped_blocks:
            if len(group) <= max_blocks_per_time:
                final_blocks.extend(group)
            else:
                group.sort(key=lambda b: b.get('offset', float('inf')))
                kept_blocks = group[:max_blocks_per_time]
                discarded_blocks = group[max_blocks_per_time:]
                for block in discarded_blocks:
                    orig_t = float(block.get("original_time", block.get("time", 0.0)))
                    block["time"] = round(orig_t, TIME_DECIMALS)
                final_blocks.extend(kept_blocks)
                final_blocks.extend(discarded_blocks)
        self.blocks = sorted(final_blocks, key=lambda x: float(x.get("time", 0.0)))
        for i in range(1, len(self.blocks)):
            prev = self.blocks[i - 1]
            cur = self.blocks[i]
            prev_t = float(prev.get("time", 0.0))
            cur_t = float(cur.get("time", 0.0))
            if round(cur_t - prev_t, TIME_DECIMALS) < MIN_GAP:
                new_t = round(prev_t + MIN_GAP, TIME_DECIMALS)
                cur["time"] = new_t
                if "original_time" in cur:
                    cur["offset"] = round(abs(cur["time"] - float(cur["original_time"])), TIME_DECIMALS)
        aligned.extend(self.blocks)

    def _estimate_difficulty(self):
        """Estimate difficulty based on block density."""
        if not self.blocks:
            return "Unknown"
        block_times = sorted([b['time'] for b in self.blocks])
        if len(block_times) < 2:
            return "Easy"
        start_time = block_times[0]
        end_time = block_times[-1]
        duration = end_time - start_time
        if duration <= 0:
            return "Unknown"
        density = len(self.blocks) / duration
        if density < 1.5:
            return "Easy"
        elif density < 3.5:
            return "Normal"
        elif density < 6:
            return "Hard"
        else:
            return "Expert"

    def _fix_repetition_and_spacing(self):
        """Prevent repetitive directions or colors and enforce minimum time gaps between blocks."""
        for i in range(len(self.blocks) - 2):
            b1, b2, b3 = self.blocks[i:i+3]
            if b1.get("type") == b2.get("type") == b3.get("type") == "note":
                if b1["direction"] == b2["direction"] == b3["direction"]:
                    choices = [d for d in range(9) if d not in (b1["direction"], b2["direction"])]
                    if choices:
                        b3["direction"] = random.choice(choices)
                    else:
                        b3["direction"] = (b3["direction"] + 1) % 9
                    if round(b3["time"] - b2["time"], self.TIME_DECIMALS) < self.MIN_GAP:
                        b3["time"] = round(b2["time"] + self.MIN_GAP, self.TIME_DECIMALS)
                if b1["color"] == b2["color"] == b3["color"]:
                    b3["color"] = 1 - b3["color"]
                    if round(b3["time"] - b2["time"], self.TIME_DECIMALS) < self.MIN_GAP:
                        b3["time"] = round(b2["time"] + self.MIN_GAP, self.TIME_DECIMALS)
            if round(b2["time"] - b1["time"], self.TIME_DECIMALS) < self.MIN_GAP:
                b2["time"] = round(b1["time"] + self.MIN_GAP, self.TIME_DECIMALS)
            if round(b3["time"] - b2["time"], self.TIME_DECIMALS) < self.MIN_GAP:
                b3["time"] = round(b2["time"] + self.MIN_GAP, self.TIME_DECIMALS)

    def _enforce_global_min_gap(self, min_gap=None):
        """Ensure all blocks maintain a minimum time gap."""
        if min_gap is None:
            min_gap = self.MIN_GAP
        self.blocks.sort(key=lambda x: x["time"])
        for i in range(1, len(self.blocks)):
            gap = round(self.blocks[i]["time"] - self.blocks[i-1]["time"], self.TIME_DECIMALS)
            if gap < min_gap:
                self.blocks[i]["time"] = round(self.blocks[i-1]["time"] + min_gap, self.TIME_DECIMALS)

    def generate(self):
        """Generate blocks (notes, bombs, walls), align them, and finalize with difficulty estimation."""
        self.blocks = []
        if not self.notes:
            print(f"⚠️ No MIDI notes loaded for {self.filename}. Skipping block generation.")
            return
        self._generate_notes()
        self._generate_bombs()
        self._generate_walls()
        self._align_blocks_to_beats()
        self._fix_repetition_and_spacing()
        self._enforce_global_min_gap()
        self._finalize_blocks()
        self.difficulty = self._estimate_difficulty()
        colors = [b['color'] for b in self.blocks if b['type'] == 'note']
        if 0 not in colors or 1 not in colors:
            dominant_color = 0 if colors.count(0) > colors.count(1) else 1
            target_color = 1 - dominant_color
            for b in self.blocks:
                if b['type'] == 'note' and b['color'] == dominant_color:
                    b['color'] = target_color
                    if dominant_color == 0 and b['direction'] == 2: b['direction'] = 3
                    elif dominant_color == 1 and b['direction'] == 3: b['direction'] = 2
                    break

    def _finalize_blocks(self):
        """Sort blocks and ensure consistent time rounding with minimum gaps."""
        if not self.blocks:
            return
        self.blocks.sort(key=lambda b: b["time"])
        last_time = None
        for i, block in enumerate(self.blocks):
            t = float(block.get("time", 0.0))
            if last_time is not None and round(t - last_time, self.TIME_DECIMALS) < self.MIN_GAP:
                t = round(last_time + self.MIN_GAP, self.TIME_DECIMALS)
            else:
                t = round(t, self.TIME_DECIMALS)
            block["time"] = t
            last_time = t

    def export(self, out_path="blocks_output.json"):
        """Export generated blocks to a JSON file with metadata."""
        if hasattr(self, "_finalize_blocks"):
            self._finalize_blocks()
        for b in self.blocks:
            if "time" in b:
                b["time"] = round(float(b["time"]), self.TIME_DECIMALS)
        output_data = {
            "audio_file": os.path.basename(self.audio_path),
            "midi_file": os.path.basename(self.midi_path),
            "tempo": self.tempo,
            "difficulty": self.difficulty,
            "total_blocks": len(self.blocks),
            "blocks": self.blocks
        }
        try:
            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Block data exported to {out_path}")
            return output_data
        except IOError as e:
            print(f"❌ Error exporting data to {out_path}: {e}")
            return None

    def plot_blocks(self, window=30):
        """Visualize a timeline of notes, bombs, and walls."""
        plt.figure(figsize=(12, 4))
        note_times = [b["time"] for b in self.blocks if b["type"] == "note"]
        bomb_times = [b["time"] for b in self.blocks if b["type"] == "bomb"]
        wall_times = [b["time"] for b in self.blocks if b["type"] == "wall"]
        plt.vlines(note_times, 0, 1, color='green', label='Notes', alpha=0.6)
        plt.vlines(bomb_times, 0, 1, color='red', label='Bombs', alpha=0.5)
        plt.vlines(wall_times, 0, 1, color='blue', label='Walls', alpha=0.4)
        plt.xlim(0, window)
        plt.title(f"Block Timeline: {self.filename}")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close('all')

    def summary(self):
        """Print a summary of the generated blocks and metadata."""
        print(f"📂 Audio: {os.path.basename(self.audio_path)}")
        print(f"🎼 MIDI : {os.path.basename(self.midi_path)}")
        print(f"🎵 Tempo: {self.tempo:.2f} BPM")
        print(f"🔥 Estimated Difficulty: {self.difficulty}")
        print(f"🔒 Strict Alignment: {self.strict_alignment}")
        print(f"🧱 Total Blocks: {len(self.blocks)}")
        type_counts = {t:0 for t in ['note', 'bomb', 'wall']}
        for b in self.blocks:
            block_type = b.get('type')
            if block_type in type_counts:
                 type_counts[block_type] += 1
        for t, c in type_counts.items():
            print(f"  🔸 {t.title():<6}: {c}")

    def plot_direction_color_heatmap(self):
        """Visualize a heatmap of note block directions vs. colors."""
        notes = [b for b in self.blocks if b["type"] == "note"]
        if not notes:
            print("⚠️ No note blocks available for visualization.")
            return
        df = pd.DataFrame(notes)
        df_group = df.groupby(['color', 'direction']).size().unstack(fill_value=0)
        for d in range(9):
            if d not in df_group.columns:
                df_group[d] = 0
        df_group = df_group[sorted(df_group.columns)]
        plt.figure(figsize=(10, 4))
        sns.heatmap(
            df_group,
            annot=True,
            fmt="d",
            cmap="coolwarm",
            cbar=True,
            xticklabels=["↓", "↑", "←", "→", "↖", "↗", "↙", "↘", "•"],
            yticklabels=["Merah (Kiri)", "Biru (Kanan)"]
        )
        plt.title(f"Heatmap Direction vs Color: {self.filename}", fontsize=14, weight='bold')
        plt.xlabel("Direction")
        plt.ylabel("Color")
        plt.tight_layout()
        plt.show()
        plt.close('all')

    def plot_distribution_summary(self):
        """Visualize distributions of note block directions and colors."""
        directions = [b["direction"] for b in self.blocks if b["type"] == "note"]
        colors = [b["color"] for b in self.blocks if b["type"] == "note"]
        dir_count = Counter(directions)
        color_count = Counter(colors)
        direction_labels = {
            0: "↓", 1: "↑", 2: "←", 3: "→",
            4: "↖", 5: "↗", 6: "↙", 7: "↘", 8: "•"
        }
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar([direction_labels.get(k, k) for k in dir_count.keys()], dir_count.values(), color='skyblue')
        ax[0].set_title("Direction Distribution")
        ax[0].set_xlabel("Swing Direction")
        ax[0].set_ylabel("Block Count")
        ax[0].grid(True, linestyle='--', alpha=0.5)
        color_names = ['Merah (Kiri)', 'Biru (Kanan)']
        color_values = [color_count.get(0, 0), color_count.get(1, 0)]
        ax[1].bar(color_names, color_values, color=['red', 'blue'])
        ax[1].set_title("Color Distribution")
        ax[1].set_ylabel("Block Count")
        ax[1].grid(True, linestyle='--', alpha=0.5)
        plt.suptitle(f"Block Distribution: {self.filename}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
        plt.close('all')

    def collect_json_results(self):
        """Return JSON data for the generated blocks and metadata."""
        return {
            "audio_file": os.path.basename(self.audio_path),
            "midi_file": os.path.basename(self.midi_path),
            "tempo": self.tempo,
            "difficulty": self.difficulty,
            "total_blocks": len(self.blocks),
            "blocks": self.blocks
        }

    @staticmethod
    def display_all_results(all_results):
        """Display a tabulated summary of JSON results for multiple songs and save to file."""
        table_data = []
        for result in all_results:
            type_counts = Counter(b["type"] for b in result["blocks"])
            block_times = sorted([b["time"] for b in result["blocks"]])
            time_gaps = np.diff(block_times) if len(block_times) > 1 else np.array([])
            min_gap = np.min(time_gaps) if time_gaps.size > 0 else float('inf')
            table_data.append([
                result["audio_file"],
                result["tempo"],
                result["difficulty"],
                result["total_blocks"],
                type_counts.get("note", 0),
                type_counts.get("bomb", 0),
                type_counts.get("wall", 0),
                f"{min_gap:.3f}" if min_gap != float('inf') else "-"
            ])
        headers = ["Song", "Tempo (BPM)", "Difficulty", "Total Blocks", "Notes", "Bombs", "Walls", "Min Time Gap (s)"]
        print("\n📊 Summary of JSON Results for All Songs:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        try:
            with open("all_songs_blocks.json", "w") as f:
                json.dump(all_results, f, indent=2)
            print("✅ All JSON results saved to 'all_songs_blocks.json'")
        except IOError as e:
            print(f"❌ Error saving JSON results: {e}")

    def cleanup(self):
        """Clear memory by resetting attributes and triggering garbage collection."""
        self.blocks = []
        self.notes = []
        self.midi = None
        self.activations = None
        del self.blocks, self.notes, self.midi, self.activations
        gc.collect()

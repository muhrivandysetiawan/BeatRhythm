class BlockDesignAnalyzerPro:
    """
    BlockDesignAnalyzerPro
    ----------------------
    A professional analyzer for rhythm game block designs stored in JSON files.
    It loads multiple JSON files, optionally auto-fixes missing fields, computes
    summaries, detects anomalies, generates formatted tables, and exports reports.

    Features:
      - Auto-fix for incomplete block data (walls, notes, bombs)
      - Statistical summaries per song (tempo, block counts, gaps)
      - Anomaly detection (tight gaps, early obstacles, imbalances, repetitions)
      - Tabular output and optional CSV/JSON exports
    """

    def __init__(self, json_files):
        """
        Initialize the analyzer with a list of JSON file paths.

        Args:
            json_files (list): List of paths to JSON files containing block data.
        """
        if not json_files:
            raise ValueError("List file JSON kosong.")
        self.json_files = json_files

    def analyze_all(self, export_path=None, export_cleaned=False, cleaned_folder="cleaned_blocks"):
        """
        Analyze all JSON files, generate summaries, details, and anomalies.
        Optionally export cleaned JSONs and a combined report.

        Args:
            export_path (str, optional): Path for the combined report file.
            export_cleaned (bool): If True, auto-fix and save cleaned JSONs.
            cleaned_folder (str): Folder to save cleaned JSON files.
        """
        if export_cleaned:
            os.makedirs(cleaned_folder, exist_ok=True)

        all_summaries, all_details, all_anomalies = [], [], []

        for jf in self.json_files:
            data = self._load_json(jf)
            if export_cleaned:
                data = self._auto_fix_blocks(data)
                clean_name = os.path.join(cleaned_folder, os.path.basename(jf).replace(".json","_cleaned.json"))
                with open(clean_name, "w", encoding="utf-8") as out:
                    json.dump(data, out, indent=2)
                print(f"✅ Cleaned JSON ditulis: {clean_name}")

            summary = self._get_summary(data)
            anomalies = self._check_anomalies(data.get("blocks", []))
            table = self._get_table(data.get("blocks", []))

            all_summaries.append(summary)
            all_details.append((summary["Nama Lagu"], table))
            all_anomalies.append((summary["Nama Lagu"], anomalies))

        # Ringkasan semua lagu (single table)
        print("\n" + "="*80)
        print("📊 Ringkasan  (semua lagu)")
        print("="*80)
        print(tabulate(all_summaries, headers="keys", tablefmt="grid"))

        # Detail per lagu (tabel sesuai desain)
        for (name, table), (_, anomalies) in zip(all_details, all_anomalies):
            print("\n" + "="*80)
            print(f"🎵 Detail Block: {name}")
            print("="*80)
            print(tabulate(table, headers=self._get_headers(), tablefmt="grid"))
            if anomalies:
                print("\n⚠️ Anomali:")
                for a in anomalies:
                    print(f" - {a}")
            else:
                print("\n✅ Tidak ada anomali mencurigakan")

        if export_path:
            self._export_report(all_summaries, all_details, all_anomalies, export_path)
            print(f"\n✅ Laporan gabungan ditulis: {export_path}")

    # ---------- helpers ----------
    def _load_json(self, path):
        """Load and return the JSON data from the given file path."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _auto_fix_blocks(self, data):
        """
        Auto-fix and normalize missing fields in blocks.
        - Walls: Infer 'direction' from 'position' if missing.
        - Notes: Ensure 'color' and 'direction' defaults.
        - Bombs: Default 'position' to 'center'.
        """
        blocks = data.get("blocks", [])
        changed = False
        pos_map = {"left":"Left", "center":"Center", "right":"Right"}

        for b in blocks:
            t = b.get("type", "").lower()
            if t == "wall":
                if "direction" not in b or b.get("direction","") == "":
                    p = str(b.get("position","center")).lower()
                    b["direction"] = pos_map.get(p, p.capitalize())
                    changed = True
                # normalize duration
                if "duration" in b:
                    try:
                        b["duration"] = float(b["duration"])
                    except:
                        b["duration"] = 0.5
                        changed = True
            elif t == "note":
                if "color" not in b:
                    b["color"] = 0
                    changed = True
                if "direction" not in b:
                    b["direction"] = 8  # dot as fallback
                    changed = True
            elif t == "bomb":
                if "position" not in b:
                    b["position"] = "center"
                    changed = True

        if changed:
            data["blocks"] = blocks
        return data

    def _get_summary(self, data):
        """Compute and return a summary dictionary for the song data."""
        blocks = data.get("blocks", [])
        note_count = sum(1 for b in blocks if b.get("type")=="note")
        bomb_count = sum(1 for b in blocks if b.get("type")=="bomb")
        wall_count = sum(1 for b in blocks if b.get("type")=="wall")
        duration = max((b.get("time",0) for b in blocks), default=0)
        avg_gap = self._average_gap(blocks)
        return {
            "Nama Lagu": data.get("audio_file","Unknown"),
            "Difficulty": data.get("difficulty","Unknown"),
            "Tempo": round(float(data.get("tempo",0)),3),
            "Total Block": data.get("total_blocks", len(blocks)),
            "Durasi (detik)": round(duration,3),
            "Notes": note_count,
            "Bombs": bomb_count,
            "Walls": wall_count,
            "Rata-rata Gap": avg_gap
        }

    def _average_gap(self, blocks):
        """Calculate the average time gap between consecutive blocks."""
        times = sorted(float(b.get("time",0)) for b in blocks)
        gaps = [t2-t1 for t1,t2 in zip(times,times[1:])] if len(times)>1 else []
        return round(statistics.mean(gaps),3) if gaps else 0

    def _check_anomalies(self, blocks):
        """
        Detect and return a list of anomalies in the block sequence.
        Checks for tight gaps, early obstacles, color imbalances, and repetitions.
        """
        anomalies = []
        sb = sorted(blocks, key=lambda x: x.get("time",0))

        # 🔎 Gap terlalu rapat
        for i in range(len(sb)-1):
            gap = sb[i+1].get("time",0) - sb[i].get("time",0)
            if gap < 0.05:
                anomalies.append(f"Block terlalu rapat di {sb[i].get('time',0):.3f}s")

        # 🔎 Wall/Bomb terlalu awal
        for b in sb:
            if b.get("time",0) < 2 and b.get("type") in ["wall","bomb"]:
                anomalies.append(f"{b.get('type')} terlalu awal di {b.get('time',0):.3f}s")

        # 🔎 Color imbalance
        colors = [b.get("color") for b in sb if b.get("type")=="note"]
        if colors:
            cnt = Counter(colors)
            if len(cnt)>1 and min(cnt.values())/max(cnt.values()) < 0.5:
                anomalies.append(f"Color note tidak seimbang: {dict(cnt)}")

        # 🔎 Repetition Violates (3 blok note berturut-turut sama)
        for i in range(len(sb)-2):
            b1, b2, b3 = sb[i:i+3]
            if all(b.get("type")=="note" for b in (b1,b2,b3)):
                # cek arah
                if b1.get("direction") == b2.get("direction") == b3.get("direction"):
                    anomalies.append(f"Arah repetitif 3x berturut-turut di sekitar {b1.get('time',0):.3f}s")
                # cek warna
                if b1.get("color") == b2.get("color") == b3.get("color"):
                    anomalies.append(f"Warna repetitif 3x berturut-turut di sekitar {b1.get('time',0):.3f}s")

        return anomalies

    def _get_table(self, blocks):
        """
        Generate a tabular representation of sorted blocks with formatted columns.
        Maps direction codes to symbols and colors to labels.
        """
        direction_labels = {
            0: "↓", 1: "↑", 2: "←", 3: "→",
            4: "↖", 5: "↗", 6: "↙", 7: "↘", 8: "•"
        }
        table = []
        for i, b in enumerate(sorted(blocks, key=lambda x: x.get("time", 0)), start=1):
            t = b.get("type", "").lower()
            block_type = t.capitalize()
            direction_val = b.get("direction", None)
            direction = ""
            if isinstance(direction_val, (int, float)):
                direction_val = int(direction_val)
                direction = direction_labels.get(direction_val, "")
            elif isinstance(direction_val, str) and direction_val.isdigit():
                 direction_val = int(direction_val)
                 direction = direction_labels.get(direction_val, "")
            color_val = b.get("color", "")
            if color_val == 0:
                color = "Red"
            elif color_val == 1:
                color = "Blue"
            else:
                color = (str(color_val) if color_val != "" else "")

            duration = b.get("duration", "") if t == "wall" else ""
            position = b.get("position", "") if t == "bomb" else ""

            table.append([
                i,
                round(float(b.get("time", 0)), 3),
                block_type,
                direction,
                color,
                duration,
                position
            ])
        return table

    def _get_headers(self):
        """Return the standard column headers for block tables."""
        return [
            "No",
            "Time",
            "Type",
            "Direction",
            "Color",
            "Duration",
            "Position"
        ]

    def _export_report(self, summaries, details, anomalies, export_path):
        """
        Export the full analysis (summaries, details, anomalies) to a text file.
        """
        with open(export_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n📊 Ringkasan\n" + "="*80 + "\n")
            f.write(tabulate(summaries, headers="keys", tablefmt="grid") + "\n\n")
            for (name, table), (_, anom) in zip(details, anomalies):
                f.write("="*80 + f"\n🎵 Detail Block: {name}\n" + "="*80 + "\n")
                f.write(tabulate(table, headers=self._get_headers(), tablefmt="grid") + "\n")
                if anom:
                    f.write("\n⚠️ Anomali:\n")
                    for a in anom:
                        f.write(f" - {a}\n")
                else:
                    f.write("\n✅ Tidak ada anomali mencurigakan\n")
                f.write("\n\n")

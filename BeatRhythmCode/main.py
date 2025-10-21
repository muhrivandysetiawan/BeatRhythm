# ------------------------------------------------------
# BeatRhyhtmDataLoader
# ------------------------------------------------------

loader = BeatRhyhtmDataLoader(verbose=True)
print(loader.song_names)
print(loader.audio_files)
print(loader.midi_files)
loader.validate_files()

pairs = loader.get_pairs()
for wav, midi in pairs:
    print(f"✅ {os.path.basename(wav)} matched with {os.path.basename(midi)}")
    print(f"🎵 {os.path.basename(wav)} ⇄ {os.path.basename(midi)}")

audio_files = loader.audio_files
lagu_list   = loader.audio_files
midi_files  = loader.midi_files

# ------------------------------------------------------
# LibrosaAudioProcessor
# ------------------------------------------------------

processor = LibrosaAudioProcessor(sr=44100, use_cache=True, max_workers=8)
processed_audio_data = processor.process_files(lagu_list)
processor.summary()
processor.export_to_json()

# ------------------------------------------------------
# AudioFeatureExtractor
# ------------------------------------------------------

dataset = []

for path in lagu_list:
    print(f"\n📦 Ekstrak fitur: {os.path.basename(path)}")
    extractor = AudioFeatureExtractor(path)
    features = extractor.extract_all()
    for key, value in features.items():
        try:
            if isinstance(value, (list, np.ndarray)):
                features[key] = [float(v) for v in value]
            else:
                features[key] = float(value)
        except (ValueError, TypeError):
            pass
    dataset.append(features)

# ------------------------------------------------------
# AudioMIDIPipeline
# ------------------------------------------------------

with open("audio_features_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    pipeline = AudioMIDIPipeline(
        audio_files=audio_files,
        midi_files=midi_files,
        tolerance=0.1,
        verbose=True,
        log_file="log_pipeline.txt",
        block_dir="blocks",
        merged_file="all_blocks.json"
    )

    pipeline.run()
    pipeline.export_report("comparison_results.csv")

# ------------------------------------------------------
# BlockDesignAnalyzerPro
# ------------------------------------------------------

json_files = [
    "/content/blocks/Crystal_Fade_blocks.json",
    "/content/blocks/CyroStomp_blocks.json",
    "/content/blocks/Digital_Mirage_blocks.json",
    "/content/blocks/Echowire_blocks.json",
    "/content/blocks/Midnight_Stack_blocks.json",
    "/content/blocks/Nova_Rapture_blocks.json",
    "/content/blocks/Pulse_Horizon_blocks.json",
    "/content/blocks/Skyborn_Circuit_blocks.json",
    "/content/blocks/Synthex_Rebirth_blocks.json",
    "/content/blocks/Venom_Drive_blocks.json"
]


analyzer = BlockDesignAnalyzerPro(json_files)
analyzer.analyze_all(export_path="BlockDesignPro_10_Song.txt", export_cleaned=True, cleaned_folder="blocks")

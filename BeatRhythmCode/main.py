


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

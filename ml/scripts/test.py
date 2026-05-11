from pathlib import Path

folder_path = Path('data/Test')
# Liczy tylko pliki (pomija katalogi)
file_count = len([f for f in folder_path.iterdir() if f.is_file()])
print(file_count)
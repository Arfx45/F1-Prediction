import kagglehub
import os
import shutil
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Download latest version
temp_path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")

# Move CSV files to data directory
for file in Path(temp_path).glob("*.csv"):
    destination = data_dir / file.name
    shutil.copy2(file, destination)
    print(f"Copied {file.name} to {destination}")

print(f"\nAll files have been copied to {data_dir.absolute()}")
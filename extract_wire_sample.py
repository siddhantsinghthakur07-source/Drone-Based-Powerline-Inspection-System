import pandas as pd
import glob
import os
import numpy as np

ROOT = r"D:\Powerline_Drone_AI\wire_geometry_dataset"
OUT = r"D:\Powerline_Drone_AI\wire_geometry_sample"

os.makedirs(OUT, exist_ok=True)

# Read the first training parquet file
files = glob.glob(os.path.join(ROOT, "train", "*.parquet"))
df = pd.read_parquet(files[0])

row = df.iloc[0]

# Save image
image = row["image"]
image_bytes = image["bytes"]
filename = row["file_name"]

image_path = os.path.join(OUT, filename)

with open(image_path, "wb") as f:
    f.write(image_bytes)

# Extract wire annotations
annotations = row["annotations"]

wire_data = []

for category, segmentation, is_line in zip(
    annotations["category_name"],
    annotations["segmentation"],
    annotations["is_line"]
):
    if category in ["conductor", "shield_wire"]:
        wire_data.append({
            "category": category,
            "is_line": bool(is_line),
            "segmentation": np.asarray(segmentation).tolist()
        })

# Save geometry information
geometry_file = os.path.join(OUT, "wire_geometry.txt")

with open(geometry_file, "w") as f:
    for i, wire in enumerate(wire_data, 1):
        f.write(f"Wire {i}\n")
        f.write(f"Category: {wire['category']}\n")
        f.write(f"Is line: {wire['is_line']}\n")
        f.write(f"Segmentation: {wire['segmentation']}\n")
        f.write("-" * 60 + "\n")

print("SAMPLE EXTRACTION COMPLETE")
print("Image:", image_path)
print("Geometry:", geometry_file)
print("Wire annotations:", len(wire_data))
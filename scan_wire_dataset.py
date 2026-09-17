import pandas as pd
import glob
import os
import numpy as np

root = r"D:\Powerline_Drone_AI\wire_geometry_dataset"

total_images = 0
conductor_count = 0
shield_count = 0
file_count = 0

print("Scanning dataset...")
print("This may take several minutes.")
print()

for split in ["train", "val", "test"]:
    files = glob.glob(os.path.join(root, split, "*.parquet"))
    print(f"{split}: {len(files)} parquet files")

    for f in files:
        file_count += 1
        df = pd.read_parquet(f)

        total_images += len(df)

        for annotations in df["annotations"]:
            categories = np.array(annotations["category_name"])

            conductor_count += int(np.sum(categories == "conductor"))
            shield_count += int(np.sum(categories == "shield_wire"))

print()
print("==============================")
print("DATASET SCAN COMPLETE")
print("==============================")
print("Parquet files:", file_count)
print("TOTAL IMAGES:", total_images)
print("CONDUCTOR ANNOTATIONS:", conductor_count)
print("SHIELD WIRE ANNOTATIONS:", shield_count)
print("TOTAL WIRE ANNOTATIONS:", conductor_count + shield_count)

import pandas as pd
import glob
import os
import json
import numpy as np

SOURCE = r"D:\Powerline_Drone_AI\wire_geometry_dataset"
OUTPUT = r"D:\Powerline_Drone_AI\wire_geometry_training"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, split, "labels"), exist_ok=True)

print("Starting full wire-geometry extraction...")
print()

total = 0
wire_annotations = 0

for split in ["train", "val", "test"]:

    parquet_files = glob.glob(
        os.path.join(SOURCE, split, "*.parquet")
    )

    print(f"{split}: {len(parquet_files)} parquet files")

    for file_index, parquet_file in enumerate(parquet_files, 1):

        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():

            image_id = int(row["image_id"])
            original_name = str(row["file_name"])

            image_data = row["image"]
            image_bytes = image_data["bytes"]

            # Use image ID to guarantee unique filenames
            extension = os.path.splitext(original_name)[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                extension = ".jpg"

            output_name = f"{image_id:06d}{extension}"

            image_path = os.path.join(
                OUTPUT, split, "images", output_name
            )

            label_path = os.path.join(
                OUTPUT, split, "labels",
                f"{image_id:06d}.json"
            )

            # Save image
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            annotations = row["annotations"]

            wire_objects = []

            for category, segmentation, is_line in zip(
                annotations["category_name"],
                annotations["segmentation"],
                annotations["is_line"]
            ):

                if category not in ["conductor", "shield_wire"]:
                    continue

                segmentation = np.asarray(
                    segmentation,
                    dtype=float
                ).tolist()

                wire_objects.append({
                    "category": str(category),
                    "is_line": bool(is_line),
                    "segmentation": segmentation
                })

            wire_annotations += len(wire_objects)

            # Save geometry
            geometry = {
                "image_id": image_id,
                "file_name": original_name,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "wires": wire_objects
            }

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as label_file:

                json.dump(
                    geometry,
                    label_file,
                    indent=2
                )

            total += 1

        print(
            f"  processed {file_index}/{len(parquet_files)}"
        )

print()
print("==============================")
print("EXTRACTION COMPLETE")
print("==============================")
print("Images extracted:", total)
print("Wire annotations:", wire_annotations)
print("Output:", OUTPUT)
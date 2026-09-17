import cv2
import json
import os
import glob
import numpy as np

SOURCE = r"D:\Powerline_Drone_AI\wire_geometry_training"
OUTPUT = r"D:\Powerline_Drone_AI\wire_segmentation_dataset"

# Artificial thickness used to convert centerlines into segmentation regions
LINE_THICKNESS = 8

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, split, "labels"), exist_ok=True)

print("======================================")
print("FULL WIRE SEGMENTATION CONVERSION")
print("======================================")
print("Source:", SOURCE)
print("Output:", OUTPUT)
print("Line thickness:", LINE_THICKNESS)
print()

total_images = 0
total_wires = 0

for split in ["train", "val", "test"]:

    label_files = glob.glob(
        os.path.join(SOURCE, split, "labels", "*.json")
    )

    print(f"{split}: {len(label_files)} images")

    for index, label_file in enumerate(label_files, 1):

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_id = int(data["image_id"])
            width = int(data["width"])
            height = int(data["height"])
            wires = data["wires"]

            # Find original extracted image
            image_files = glob.glob(
                os.path.join(
                    SOURCE,
                    split,
                    "images",
                    f"{image_id:06d}.*"
                )
            )

            if not image_files:
                print("WARNING: Image missing:", image_id)
                continue

            image_file = image_files[0]

            image = cv2.imread(image_file)

            if image is None:
                print("WARNING: Could not read:", image_file)
                continue

            extension = os.path.splitext(image_file)[1].lower()

            if extension not in [".jpg", ".jpeg", ".png"]:
                extension = ".jpg"

            output_name = f"{image_id:06d}{extension}"

            output_image = os.path.join(
                OUTPUT,
                split,
                "images",
                output_name
            )

            output_label = os.path.join(
                OUTPUT,
                split,
                "labels",
                f"{image_id:06d}.txt"
            )

            # Copy image
            cv2.imwrite(output_image, image)

            yolo_lines = []

            for wire in wires:

                category = wire["category"]

                if category == "conductor":
                    class_id = 0

                elif category == "shield_wire":
                    class_id = 1

                else:
                    continue

                points = np.asarray(
                    wire["segmentation"],
                    dtype=np.float32
                ).reshape(-1, 2)

                if len(points) < 2:
                    continue

                # Keep coordinates inside image
                points[:, 0] = np.clip(
                    points[:, 0], 0, width - 1
                )

                points[:, 1] = np.clip(
                    points[:, 1], 0, height - 1
                )

                points_int = np.round(
                    points
                ).astype(np.int32)

                # Create thin wire mask
                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8
                )

                cv2.polylines(
                    mask,
                    [points_int],
                    False,
                    255,
                    thickness=LINE_THICKNESS,
                    lineType=cv2.LINE_AA
                )

                # Extract polygon around line
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:

                    if len(contour) < 3:
                        continue

                    polygon = contour.reshape(
                        -1, 2
                    ).astype(np.float32)

                    # Normalize coordinates
                    polygon[:, 0] /= width
                    polygon[:, 1] /= height

                    polygon = np.clip(
                        polygon,
                        0.0,
                        1.0
                    )

                    # Need at least 3 polygon points
                    if len(polygon) < 3:
                        continue

                    values = [str(class_id)]

                    for x, y in polygon:
                        values.append(f"{x:.6f}")
                        values.append(f"{y:.6f}")

                    yolo_lines.append(
                        " ".join(values)
                    )

                    total_wires += 1

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as f:
                f.write("\n".join(yolo_lines))

            total_images += 1

        except Exception as e:
            print()
            print("ERROR processing:", label_file)
            print(e)
            print()
            continue

        # Progress every 100 images
        if index % 100 == 0 or index == len(label_files):
            print(
                f"  processed {index}/{len(label_files)}"
            )

print()
print("======================================")
print("CONVERSION COMPLETE")
print("======================================")
print("Images converted:", total_images)
print("Wire segmentation objects:", total_wires)
print("Output:", OUTPUT)
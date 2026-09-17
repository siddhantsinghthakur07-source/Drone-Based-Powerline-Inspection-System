import cv2
import json
import os
import glob
import numpy as np

SOURCE = r"D:\Powerline_Drone_AI\wire_geometry_training\train"
OUTPUT = r"D:\Powerline_Drone_AI\wire_segmentation_sample"

os.makedirs(os.path.join(OUTPUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "labels"), exist_ok=True)

# Get first JSON label
label_files = glob.glob(
    os.path.join(SOURCE, "labels", "*.json")
)

label_file = label_files[0]

with open(label_file, "r", encoding="utf-8") as f:
    data = json.load(f)

image_id = data["image_id"]
width = data["width"]
height = data["height"]
wires = data["wires"]

# Find corresponding image
image_files = glob.glob(
    os.path.join(SOURCE, "images", f"{image_id:06d}.*")
)

if not image_files:
    raise FileNotFoundError("Corresponding image not found")

image_file = image_files[0]

# Copy image
image = cv2.imread(image_file)

if image is None:
    raise RuntimeError("Could not read image")

output_image = os.path.join(
    OUTPUT,
    "images",
    os.path.basename(image_file)
)

cv2.imwrite(output_image, image)

yolo_lines = []

# Thickness of artificial segmentation line
LINE_THICKNESS = 8

for wire in wires:

    category = wire["category"]

    if category == "conductor":
        class_id = 0

    elif category == "shield_wire":
        class_id = 1

    else:
        continue

    points = np.array(
        wire["segmentation"],
        dtype=np.float32
    ).reshape(-1, 2)

    if len(points) < 2:
        continue

    points_int = np.round(points).astype(np.int32)

    # Create mask for this individual wire
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

    # Find polygon around the thin line
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if len(contour) < 3:
            continue

        polygon = contour.reshape(-1, 2).astype(float)

        # Normalize coordinates for YOLO
        polygon[:, 0] /= width
        polygon[:, 1] /= height

        polygon = np.clip(polygon, 0.0, 1.0)

        values = [str(class_id)]

        for x, y in polygon:
            values.append(f"{x:.6f}")
            values.append(f"{y:.6f}")

        yolo_lines.append(" ".join(values))

# Save YOLO segmentation label
output_label = os.path.join(
    OUTPUT,
    "labels",
    f"{image_id:06d}.txt"
)

with open(output_label, "w") as f:
    f.write("\n".join(yolo_lines))

print("SAMPLE CONVERSION COMPLETE")
print("Image:", output_image)
print("Label:", output_label)
print("Wire objects:", len(yolo_lines))
print("Line thickness:", LINE_THICKNESS)
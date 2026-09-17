import cv2
import glob
import os
import numpy as np

SOURCE = r"D:\Powerline_Drone_AI\wire_segmentation_sample"

image_file = glob.glob(
    os.path.join(SOURCE, "images", "*")
)[0]

label_file = glob.glob(
    os.path.join(SOURCE, "labels", "*.txt")
)[0]

image = cv2.imread(image_file)

if image is None:
    raise RuntimeError("Could not load image")

height, width = image.shape[:2]

with open(label_file, "r") as f:
    lines = f.readlines()

for line in lines:

    values = line.strip().split()

    if len(values) < 7:
        continue

    class_id = int(values[0])

    coords = np.array(
        [float(x) for x in values[1:]],
        dtype=np.float32
    ).reshape(-1, 2)

    points = np.zeros_like(coords)

    points[:, 0] = coords[:, 0] * width
    points[:, 1] = coords[:, 1] * height

    points = np.round(points).astype(np.int32)

    if class_id == 0:
        color = (0, 255, 0)
        name = "CONDUCTOR"
    else:
        color = (0, 0, 255)
        name = "SHIELD WIRE"

    cv2.polylines(
        image,
        [points],
        True,
        color,
        2
    )

    x, y = points[0]

    cv2.putText(
        image,
        name,
        (x, max(y - 5, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

output = os.path.join(
    SOURCE,
    "wire_overlay.jpg"
)

cv2.imwrite(output, image)

print("VISUALIZATION COMPLETE")
print("Output:", output)
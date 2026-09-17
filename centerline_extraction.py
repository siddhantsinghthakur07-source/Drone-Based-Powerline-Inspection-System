from ultralytics import YOLO
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path
import csv

# ============================================================
# WIRE CENTERLINE EXTRACTION - CORRECTED VERSION
# ============================================================

MODEL_PATH = r"D:\Powerline_Drone_AI\training_results\wire_segmentation_rtx2050\weights\best.pt"

IMAGE_PATH = r"D:\Powerline_Drone_AI\test_sample\006836.jpg"

OUTPUT_DIR = Path(
    r"D:\Powerline_Drone_AI\centerline_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25

# ============================================================
# START
# ============================================================

print()
print("==========================================")
print("WIRE CENTERLINE EXTRACTION")
print("CORRECTED ORIGINAL-COORDINATE VERSION")
print("==========================================")
print()

print(f"Model : {MODEL_PATH}")
print(f"Image : {IMAGE_PATH}")
print()

# ============================================================
# CHECK FILES
# ============================================================

if not Path(MODEL_PATH).exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not Path(IMAGE_PATH).exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()

# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise RuntimeError(
        "Could not read the input image."
    )

height, width = image.shape[:2]

print(
    f"Original image size: "
    f"{width} x {height}"
)

print()

# ============================================================
# RUN YOLO
# ============================================================

print("Running segmentation...")

results = model.predict(

    source=IMAGE_PATH,

    conf=CONFIDENCE,

    imgsz=640,

    retina_masks=True,

    verbose=False,

    save=False
)

result = results[0]

# ============================================================
# CHECK MASKS
# ============================================================

if result.masks is None:

    print(
        "No segmentation masks detected."
    )

    raise SystemExit

# IMPORTANT:
# masks.xy contains polygon coordinates
# already mapped to the ORIGINAL image.

polygons = result.masks.xy

class_ids = (
    result.boxes.cls
    .cpu()
    .numpy()
    .astype(int)
)

confidences = (
    result.boxes.conf
    .cpu()
    .numpy()
)

print(
    f"Masks found: {len(polygons)}"
)

print()

# ============================================================
# OUTPUT IMAGE
# ============================================================

output = image.copy()

# Different colors for each wire

colors = [

    (0, 0, 255),       # Red

    (0, 255, 0),       # Green

    (255, 0, 0),       # Blue

    (0, 255, 255),     # Yellow

    (255, 0, 255),     # Magenta

    (255, 255, 0),     # Cyan

    (0, 128, 255),     # Orange

    (128, 0, 255)      # Purple
]

# ============================================================
# CSV
# ============================================================

csv_rows = []

wire_number = 0

# ============================================================
# PROCESS EACH PREDICTION
# ============================================================

for i, polygon in enumerate(polygons):

    class_id = class_ids[i]

    confidence = float(
        confidences[i]
    )

    class_name = model.names[
        class_id
    ]

    # --------------------------------------------------------
    # CREATE FULL-SIZE MASK
    # --------------------------------------------------------

    binary_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Polygon coordinates returned by
    # Ultralytics are already in original
    # image coordinates.

    polygon_int = np.round(
        polygon
    ).astype(
        np.int32
    )

    # Fill polygon

    cv2.fillPoly(
        binary_mask,
        [polygon_int],
        255
    )

    # --------------------------------------------------------
    # OPTIONAL SMALL CLEANUP
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # SKELETONIZATION
    # --------------------------------------------------------

    skeleton = skeletonize(
        binary_mask > 0
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENT CLEANUP
    # --------------------------------------------------------

    skeleton_uint8 = (
        skeleton.astype(
            np.uint8
        )
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            skeleton_uint8,
            connectivity=8
        )
    )

    if num_labels > 1:

        component_sizes = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest_label = (
            1 +
            np.argmax(
                component_sizes
            )
        )

        clean_skeleton = (
            labels == largest_label
        )

    else:

        clean_skeleton = skeleton

    # --------------------------------------------------------
    # CENTERLINE COORDINATES
    # --------------------------------------------------------

    y_coords, x_coords = np.where(
        clean_skeleton
    )

    if len(x_coords) == 0:

        print(
            f"Prediction {i + 1}: "
            f"No centerline found."
        )

        continue

    wire_number += 1

    color = colors[
        (wire_number - 1)
        % len(colors)
    ]

    print(
        f"Wire {wire_number}: "
        f"{class_name} | "
        f"confidence={confidence:.2f} | "
        f"points={len(x_coords)}"
    )

    # --------------------------------------------------------
    # DRAW CENTERLINE
    # --------------------------------------------------------

    for x, y in zip(
        x_coords,
        y_coords
    ):

        cv2.circle(

            output,

            (
                int(x),
                int(y)
            ),

            2,

            color,

            -1
        )

    # --------------------------------------------------------
    # FIND LABEL POSITION
    # --------------------------------------------------------

    middle_index = (
        len(x_coords) // 2
    )

    label_x = int(
        x_coords[middle_index]
    )

    label_y = int(
        y_coords[middle_index]
    )

    label = (
        f"W{wire_number} "
        f"{class_name}"
    )

    cv2.putText(

        output,

        label,

        (
            label_x,
            label_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        color,

        2,

        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    mask_filename = (

        f"mask_{wire_number}_"

        f"{class_name}.png"
    )

    cv2.imwrite(

        str(
            OUTPUT_DIR /
            mask_filename
        ),

        binary_mask
    )

    # --------------------------------------------------------
    # SAVE CENTERLINE IMAGE
    # --------------------------------------------------------

    centerline_image = (
        np.zeros_like(image)
    )

    centerline_image[
        clean_skeleton
    ] = color

    centerline_filename = (

        f"centerline_{wire_number}_"

        f"{class_name}.png"
    )

    cv2.imwrite(

        str(
            OUTPUT_DIR /
            centerline_filename
        ),

        centerline_image
    )

    # --------------------------------------------------------
    # SAVE CSV DATA
    # --------------------------------------------------------

    for x, y in zip(
        x_coords,
        y_coords
    ):

        csv_rows.append([

            wire_number,

            class_name,

            confidence,

            int(x),

            int(y)
        ])

# ============================================================
# SAVE FINAL VISUALIZATION
# ============================================================

output_image = (

    OUTPUT_DIR /

    "006836_centerlines_corrected.jpg"
)

cv2.imwrite(

    str(output_image),

    output
)

# ============================================================
# SAVE CSV
# ============================================================

output_csv = (

    OUTPUT_DIR /

    "006836_centerlines_corrected.csv"
)

with open(

    output_csv,

    "w",

    newline="",

    encoding="utf-8"

) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([

        "wire_id",

        "class",

        "confidence",

        "x",

        "y"
    ])

    writer.writerows(
        csv_rows
    )

# ============================================================
# SUMMARY
# ============================================================

print()

print("==========================================")

print("CENTERLINE EXTRACTION COMPLETE")

print("==========================================")

print()

print(
    f"Wires processed : "
    f"{wire_number}"
)

print()

print(
    f"Visualization:"
)

print(
    output_image
)

print()

print(
    f"CSV:"
)

print(
    output_csv
)

print()

print(
    "IMPORTANT:"
)

print(
    "Centerlines were generated "
    "from masks.xy in original "
    "image coordinates."
)

print()
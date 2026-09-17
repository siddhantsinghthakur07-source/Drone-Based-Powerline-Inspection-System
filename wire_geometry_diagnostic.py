from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd


# ============================================================
# WIRE SEGMENTATION INFERENCE DIAGNOSTIC
# ============================================================

MODEL_PATH = r"D:\Powerline_Drone_AI\training_results\wire_segmentation_rtx2050\weights\best.pt"

IMAGE_DIR = Path(
    r"D:\Powerline_Drone_AI\test_sample"
)

OUTPUT_DIR = Path(
    r"D:\Powerline_Drone_AI\wire_diagnostic_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INFERENCE SETTINGS
# ============================================================

CONFIDENCE = 0.10

IMAGE_SIZE = 960


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 65)
print("WIRE SEGMENTATION INFERENCE DIAGNOSTIC")
print("=" * 65)
print()

print("Model:")
print(MODEL_PATH)

print()

print("Image directory:")
print(IMAGE_DIR)

print()

print(
    f"Inference image size : {IMAGE_SIZE}"
)

print(
    f"Confidence threshold : {CONFIDENCE}"
)

print()


if not Path(MODEL_PATH).exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("Model loaded successfully.")

print()


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

if not images:

    images = sorted(
        IMAGE_DIR.glob("*.jpeg")
    )

if not images:

    raise FileNotFoundError(
        f"No images found in:\n{IMAGE_DIR}"
    )


print(
    f"Images found: {len(images)}"
)

print()


# ============================================================
# RESULTS TABLE
# ============================================================

diagnostic_results = []


# ============================================================
# PROCESS IMAGES
# ============================================================

for number, image_path in enumerate(
    images,
    start=1
):

    print("=" * 65)

    print(
        f"[{number}/{len(images)}] "
        f"{image_path.name}"
    )

    print("=" * 65)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Could not read image."
        )

        continue


    height, width = image.shape[:2]

    print(
        f"Original size: "
        f"{width} x {height}"
    )

    print()


    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    result = model.predict(

        source=str(image_path),

        conf=CONFIDENCE,

        imgsz=IMAGE_SIZE,

        retina_masks=True,

        verbose=False,

        save=False

    )[0]


    # ========================================================
    # NO MASKS
    # ========================================================

    if result.masks is None:

        print(
            "NO MASKS DETECTED"
        )

        diagnostic_results.append({

            "image":
                image_path.name,

            "masks":
                0,

            "conductor":
                0,

            "shield_wire":
                0,

            "other":
                0,

            "status":
                "No masks"

        })

        print()

        continue


    # ========================================================
    # EXTRACT PREDICTIONS
    # ========================================================

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


    total_masks = len(
        class_ids
    )


    conductor_count = 0

    shield_count = 0

    other_count = 0


    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualization = image.copy()


    # ========================================================
    # PROCESS EACH DETECTION
    # ========================================================

    for detection_number in range(
        total_masks
    ):

        class_id = class_ids[
            detection_number
        ]

        confidence = float(
            confidences[
                detection_number
            ]
        )


        class_name = model.names[
            class_id
        ]


        print(

            f"Detection "
            f"{detection_number + 1}: "

            f"{class_name} "

            f"confidence="
            f"{confidence:.3f}"

        )


        # ----------------------------------------------------
        # COUNT CLASSES
        # ----------------------------------------------------

        if class_name.lower() == "conductor":

            conductor_count += 1

            color = (
                255,
                0,
                0
            )


        elif class_name.lower() == "shield_wire":

            shield_count += 1

            color = (
                0,
                255,
                255
            )


        else:

            other_count += 1

            color = (
                0,
                255,
                0
            )


        # ----------------------------------------------------
        # DRAW SEGMENTATION MASK
        # ----------------------------------------------------

        polygon = (
            result.masks.xy[
                detection_number
            ]
        )


        polygon_int = (
            polygon
            .round()
            .astype(
                "int32"
            )
        )


        cv2.polylines(

            visualization,

            [polygon_int],

            True,

            color,

            3

        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        x, y = polygon_int[0]


        cv2.putText(

            visualization,

            f"{class_name} "
            f"{confidence:.2f}",

            (
                int(x),
                int(y)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # PRINT IMAGE SUMMARY
    # ========================================================

    print()

    print(
        f"Total masks     : {total_masks}"
    )

    print(
        f"Conductors      : {conductor_count}"
    )

    print(
        f"Shield wires    : {shield_count}"
    )

    print(
        f"Other           : {other_count}"
    )


    if conductor_count >= 2:

        status = "Good"

        print(
            "STATUS: "
            "Enough conductors for geometry"
        )

    elif conductor_count == 1:

        status = "One conductor"

        print(
            "STATUS: "
            "Only one conductor detected"
        )

    else:

        status = "No conductor"

        print(
            "STATUS: "
            "No conductor detected"
        )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    output_image = (

        OUTPUT_DIR /

        f"{image_path.stem}_diagnostic.jpg"

    )


    cv2.imwrite(

        str(output_image),

        visualization

    )


    print()

    print(
        "Visualization saved:"
    )

    print(
        output_image
    )

    print()


    # ========================================================
    # SAVE DATA
    # ========================================================

    diagnostic_results.append({

        "image":
            image_path.name,

        "width":
            width,

        "height":
            height,

        "masks":
            total_masks,

        "conductor":
            conductor_count,

        "shield_wire":
            shield_count,

        "other":
            other_count,

        "status":
            status

    })


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    diagnostic_results
)


csv_path = (

    OUTPUT_DIR /

    "diagnostic_summary.csv"

)


df.to_csv(

    csv_path,

    index=False

)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 65)
print("DIAGNOSTIC COMPLETE")
print("=" * 65)
print()

print(
    "Summary CSV:"
)

print(
    csv_path
)

print()

print(
    "Diagnostic images:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "IMPORTANT:"
)

print(
    "This test does NOT retrain the model."
)

print(
    "It only tests the existing best.pt "
    "at higher inference resolution."
)

print()
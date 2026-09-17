from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# BATCH WIRE GEOMETRY EVALUATION
# ============================================================

MODEL_PATH = r"D:\Powerline_Drone_AI\training_results\wire_segmentation_rtx2050\weights\best.pt"

IMAGE_DIR = Path(
    r"D:\Powerline_Drone_AI\test_sample"
)

OUTPUT_DIR = Path(
    r"D:\Powerline_Drone_AI\batch_geometry_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFIDENCE = 0.25
IMG_SIZE = 640

NUM_SAMPLES = 400
SMOOTH_WINDOW = 21
MIN_POINTS = 30


# ============================================================
# SMOOTHING
# ============================================================

def smooth(values, window):

    if len(values) < 5:
        return values.copy()

    window = min(
        window,
        len(values)
    )

    if window % 2 == 0:
        window -= 1

    if window < 5:
        return values.copy()

    kernel = (
        np.ones(window) /
        window
    )

    padded = np.pad(
        values,
        window // 2,
        mode="edge"
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid"
    )


# ============================================================
# EXTRACT CENTERLINE
# ============================================================

def get_centerline(mask):

    skeleton = skeletonize(
        mask > 0
    )

    skeleton_uint8 = (
        skeleton.astype(np.uint8)
    )

    number, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            skeleton_uint8,
            connectivity=8
        )
    )

    if number > 1:

        sizes = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest = (
            1 +
            np.argmax(sizes)
        )

        skeleton = (
            labels == largest
        )

    return skeleton


# ============================================================
# CONVERT CENTERLINE TO SMOOTH CURVE
# ============================================================

def prepare_curve(x, y):

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    unique_x = np.unique(x)

    clean_x = []
    clean_y = []

    for value in unique_x:

        ys = y[x == value]

        clean_x.append(value)
        clean_y.append(np.median(ys))

    clean_x = np.asarray(
        clean_x,
        dtype=float
    )

    clean_y = np.asarray(
        clean_y,
        dtype=float
    )

    if len(clean_x) < MIN_POINTS:
        return None

    sample_x = np.linspace(
        clean_x.min(),
        clean_x.max(),
        NUM_SAMPLES
    )

    sample_y = np.interp(
        sample_x,
        clean_x,
        clean_y
    )

    sample_y = smooth(
        sample_y,
        SMOOTH_WINDOW
    )

    return sample_x, sample_y


# ============================================================
# CLOSEST CENTERLINE DISTANCE
# ============================================================

def calculate_separation(
    x1,
    y1,
    x2,
    y2
):

    start = max(
        x1.min(),
        x2.min()
    )

    end = min(
        x1.max(),
        x2.max()
    )

    if end <= start:
        return None

    x = np.linspace(
        start,
        end,
        NUM_SAMPLES
    )

    ya = np.interp(
        x,
        x1,
        y1
    )

    yb = np.interp(
        x,
        x2,
        y2
    )

    # --------------------------------------------------------
    # Calculate closest distance between the two curves.
    #
    # This is more geometrically meaningful than simply
    # subtracting Y coordinates.
    # --------------------------------------------------------

    distances = []

    locations = []

    for i in range(
        len(x)
    ):

        px = x[i]
        py = ya[i]

        dx = x - px
        dy = yb - py

        distance = np.sqrt(
            dx * dx +
            dy * dy
        )

        index = np.argmin(
            distance
        )

        distances.append(
            distance[index]
        )

        locations.append(
            (
                px,
                py,
                x[index],
                yb[index]
            )
        )

    distances = np.asarray(
        distances
    )

    distances_smooth = smooth(
        distances,
        SMOOTH_WINDOW
    )

    return (
        x,
        ya,
        yb,
        distances,
        distances_smooth,
        locations
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("BATCH WIRE GEOMETRY EVALUATION")
print("=" * 60)
print()

print("Loading model...")

model = YOLO(
    MODEL_PATH
)

print("Model loaded.")
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

    raise RuntimeError(
        f"No images found in {IMAGE_DIR}"
    )

print(
    f"Images found: {len(images)}"
)

for image in images:
    print(
        f"  {image.name}"
    )

print()


# ============================================================
# RESULTS
# ============================================================

summary = []

total_conductor_pairs = 0


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_number, image_path in enumerate(
    images,
    start=1
):

    print()
    print("=" * 60)
    print(
        f"[{image_number}/{len(images)}] "
        f"{image_path.name}"
    )
    print("=" * 60)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Could not read image. Skipping."
        )

        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    result = model.predict(

        source=str(image_path),

        conf=CONFIDENCE,

        imgsz=IMG_SIZE,

        retina_masks=True,

        verbose=False,

        save=False
    )[0]

    if result.masks is None:

        print(
            "No masks detected."
        )

        summary.append({

            "image":
                image_path.name,

            "conductors":
                0,

            "pairs":
                0,

            "status":
                "No masks"

        })

        continue

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
        f"Masks detected: "
        f"{len(polygons)}"
    )

    # --------------------------------------------------------
    # CENTERLINE VISUALIZATION
    # --------------------------------------------------------

    visualization = image.copy()

    colors = [

        (0, 0, 255),

        (0, 255, 0),

        (255, 0, 0),

        (0, 255, 255),

        (255, 0, 255),

        (255, 255, 0),

        (0, 128, 255),

        (128, 0, 255)

    ]

    wires = {}

    wire_id = 0

    # --------------------------------------------------------
    # EXTRACT WIRES
    # --------------------------------------------------------

    for i, polygon in enumerate(
        polygons
    ):

        class_id = class_ids[i]

        confidence = float(
            confidences[i]
        )

        class_name = model.names[
            class_id
        ]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        polygon_int = np.round(
            polygon
        ).astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [polygon_int],
            255
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        skeleton = get_centerline(
            mask
        )

        y_pixels, x_pixels = np.where(
            skeleton
        )

        if len(x_pixels) < MIN_POINTS:
            continue

        curve = prepare_curve(
            x_pixels,
            y_pixels
        )

        if curve is None:
            continue

        curve_x, curve_y = curve

        wire_id += 1

        wires[wire_id] = {

            "class":
                class_name,

            "confidence":
                confidence,

            "x":
                curve_x,

            "y":
                curve_y

        }

        color = colors[
            (wire_id - 1)
            % len(colors)
        ]

        # Draw centerline

        for x, y in zip(
            curve_x,
            curve_y
        ):

            cv2.circle(

                visualization,

                (
                    int(x),
                    int(y)
                ),

                2,

                color,

                -1
            )

        middle = len(
            curve_x
        ) // 2

        cv2.putText(

            visualization,

            f"W{wire_id} {class_name}",

            (
                int(curve_x[middle]),
                int(curve_y[middle])
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2,

            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # CONDUCTORS
    # --------------------------------------------------------

    conductors = [

        wid

        for wid, data
        in wires.items()

        if data["class"].lower()
        == "conductor"

    ]

    print(
        f"Conductors: "
        f"{len(conductors)}"
    )

    # Save visualization

    visualization_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_centerlines.jpg"
    )

    cv2.imwrite(
        str(visualization_path),
        visualization
    )

    # --------------------------------------------------------
    # NOT ENOUGH CONDUCTORS
    # --------------------------------------------------------

    if len(conductors) < 2:

        print(
            "Less than 2 conductors. "
            "Skipping separation."
        )

        summary.append({

            "image":
                image_path.name,

            "conductors":
                len(conductors),

            "pairs":
                0,

            "status":
                "Insufficient conductors"

        })

        continue

    # --------------------------------------------------------
    # PAIRWISE SEPARATION
    # --------------------------------------------------------

    image_pairs = 0

    image_separations = []

    for a in range(
        len(conductors)
    ):

        for b in range(
            a + 1,
            len(conductors)
        ):

            id_a = conductors[a]
            id_b = conductors[b]

            wire_a = wires[id_a]
            wire_b = wires[id_b]

            result_data = calculate_separation(

                wire_a["x"],
                wire_a["y"],

                wire_b["x"],
                wire_b["y"]

            )

            if result_data is None:
                continue

            (
                x,
                ya,
                yb,
                distances,
                distances_smooth,
                locations
            ) = result_data

            mean_sep = float(
                np.mean(
                    distances_smooth
                )
            )

            median_sep = float(
                np.median(
                    distances_smooth
                )
            )

            min_index = int(
                np.argmin(
                    distances_smooth
                )
            )

            max_index = int(
                np.argmax(
                    distances_smooth
                )
            )

            min_sep = float(
                distances_smooth[
                    min_index
                ]
            )

            max_sep = float(
                distances_smooth[
                    max_index
                ]
            )

            image_pairs += 1

            total_conductor_pairs += 1

            image_separations.append(
                mean_sep
            )

            print(

                f"W{id_a}-W{id_b}: "

                f"mean={mean_sep:.2f}px | "

                f"min={min_sep:.2f}px | "

                f"max={max_sep:.2f}px"

            )

            # ------------------------------------------------
            # Save pair profile
            # ------------------------------------------------

            pair_profile = pd.DataFrame({

                "x_px":
                    x,

                "wire_1_y_px":
                    ya,

                "wire_2_y_px":
                    yb,

                "separation_px":
                    distances,

                "smoothed_separation_px":
                    distances_smooth

            })

            pair_profile_path = (

                OUTPUT_DIR /

                f"{image_path.stem}_"
                f"W{id_a}_W{id_b}_"
                f"profile.csv"

            )

            pair_profile.to_csv(

                pair_profile_path,

                index=False
            )

            # ------------------------------------------------
            # Save graph
            # ------------------------------------------------

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(

                x,

                distances,

                linewidth=1,

                label="Raw"

            )

            plt.plot(

                x,

                distances_smooth,

                linewidth=2,

                label="Smoothed"

            )

            plt.xlabel(
                "Image X Position (px)"
            )

            plt.ylabel(
                "Conductor Separation (px)"
            )

            plt.title(

                f"{image_path.name} "
                f"W{id_a}-W{id_b}"

            )

            plt.grid(
                True,
                alpha=0.3
            )

            plt.legend()

            plt.tight_layout()

            graph_path = (

                OUTPUT_DIR /

                f"{image_path.stem}_"
                f"W{id_a}_W{id_b}_"
                f"profile.png"

            )

            plt.savefig(

                graph_path,

                dpi=150

            )

            plt.close()

    # --------------------------------------------------------
    # IMAGE SUMMARY
    # --------------------------------------------------------

    if image_pairs > 0:

        summary.append({

            "image":
                image_path.name,

            "conductors":
                len(conductors),

            "pairs":
                image_pairs,

            "mean_pair_separation_px":
                float(
                    np.mean(
                        image_separations
                    )
                ),

            "status":
                "Success"

        })

    else:

        summary.append({

            "image":
                image_path.name,

            "conductors":
                len(conductors),

            "pairs":
                0,

            "status":
                "No valid pairs"

        })


# ============================================================
# SAVE MASTER CSV
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_path = (
    OUTPUT_DIR /
    "BATCH_GEOMETRY_SUMMARY.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("BATCH GEOMETRY EVALUATION COMPLETE")
print("=" * 60)
print()

print(
    f"Images processed : "
    f"{len(images)}"
)

print(
    f"Valid conductor pairs : "
    f"{total_conductor_pairs}"
)

print()

print(
    "Master summary:"
)

print(
    summary_path
)

print()

print(
    "All outputs:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "IMPORTANT:"
)

print(
    "All separation values are IMAGE-SPACE PIXELS."
)

print(
    "They are not physical distances."
)

print()
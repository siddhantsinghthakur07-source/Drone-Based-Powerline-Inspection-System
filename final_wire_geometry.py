from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# FINAL WIRE GEOMETRY V3
# ============================================================

MODEL_PATH = r"D:\Powerline_Drone_AI\training_results\wire_segmentation_rtx2050\weights\best.pt"
IMAGE_DIR = Path(r"D:\Powerline_Drone_AI\test_sample")
OUTPUT_DIR = Path(r"D:\Powerline_Drone_AI\final_geometry_results_v3")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = 960
CONF = 0.45
MIN_POINTS = 40
NUM_BINS = 500

DUPLICATE_DISTANCE = 15
MASK_IOU_THRESHOLD = 0.50


# ============================================================
# CENTERLINE
# ============================================================

def extract_centerline(mask):

    y, x = np.where(mask > 0)

    if len(x) < MIN_POINTS:
        return None

    points = np.column_stack([
        x.astype(float),
        y.astype(float)
    ])

    center = points.mean(axis=0)

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    direction = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    if direction[0] < 0:
        direction = -direction

    normal = np.array([
        -direction[1],
        direction[0]
    ])

    u = centered @ direction
    v = centered @ normal

    bins = np.linspace(
        u.min(),
        u.max(),
        NUM_BINS + 1
    )

    centers_u = []
    centers_v = []

    for i in range(NUM_BINS):

        idx = (
            (u >= bins[i]) &
            (u < bins[i + 1])
        )

        if np.sum(idx) < 2:
            continue

        centers_u.append(
            np.median(u[idx])
        )

        centers_v.append(
            np.median(v[idx])
        )

    if len(centers_u) < MIN_POINTS:
        return None

    centers_u = np.array(
        centers_u
    )

    centers_v = np.array(
        centers_v
    )

    # Smooth
    if len(centers_v) >= 11:

        kernel = np.ones(11) / 11

        centers_v = np.convolve(
            centers_v,
            kernel,
            mode="same"
        )

    centerline = (
        center
        + np.outer(centers_u, direction)
        + np.outer(centers_v, normal)
    )

    return centerline


# ============================================================
# MASK IOU
# ============================================================

def mask_iou(a, b):

    intersection = np.logical_and(
        a > 0,
        b > 0
    ).sum()

    union = np.logical_or(
        a > 0,
        b > 0
    ).sum()

    if union == 0:
        return 0

    return intersection / union


# ============================================================
# CENTERLINE OVERLAP DISTANCE
# ============================================================

def centerline_overlap_distance(a, b):

    p1 = a["centerline"]
    p2 = b["centerline"]

    x1 = p1[:, 0]
    y1 = p1[:, 1]

    x2 = p2[:, 0]
    y2 = p2[:, 1]

    xmin = max(
        x1.min(),
        x2.min()
    )

    xmax = min(
        x1.max(),
        x2.max()
    )

    if xmax <= xmin:
        return float("inf")

    samples = np.linspace(
        xmin,
        xmax,
        200
    )

    y1_interp = np.interp(
        samples,
        x1,
        y1
    )

    y2_interp = np.interp(
        samples,
        x2,
        y2
    )

    distances = np.abs(
        y1_interp - y2_interp
    )

    return float(
        np.median(distances)
    )


# ============================================================
# DUPLICATE FILTER
# ============================================================

def remove_duplicates(candidates):

    candidates.sort(
        key=lambda x:
        x["confidence"],
        reverse=True
    )

    final = []

    for candidate in candidates:

        duplicate = False

        for existing in final:

            iou = mask_iou(
                candidate["mask"],
                existing["mask"]
            )

            distance = (
                centerline_overlap_distance(
                    candidate,
                    existing
                )
            )

            if (
                iou >= MASK_IOU_THRESHOLD
                or distance < DUPLICATE_DISTANCE
            ):

                duplicate = True
                break

        if not duplicate:
            final.append(candidate)

    return final


# ============================================================
# SEPARATION
# ============================================================

def calculate_separation(w1, w2):

    p1 = w1["centerline"]
    p2 = w2["centerline"]

    x1 = p1[:, 0]
    y1 = p1[:, 1]

    x2 = p2[:, 0]
    y2 = p2[:, 1]

    xmin = max(
        x1.min(),
        x2.min()
    )

    xmax = min(
        x1.max(),
        x2.max()
    )

    if xmax <= xmin:
        return None

    x = np.linspace(
        xmin,
        xmax,
        500
    )

    y1_interp = np.interp(
        x,
        x1,
        y1
    )

    y2_interp = np.interp(
        x,
        x2,
        y2
    )

    separation = np.abs(
        y1_interp - y2_interp
    )

    return x, separation


# ============================================================
# VISUALIZATION
# ============================================================

def draw_result(
    image,
    conductors,
    path
):

    output = image.copy()

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0)
    ]

    for i, wire in enumerate(
        conductors
    ):

        color = colors[
            i % len(colors)
        ]

        pts = np.round(
            wire["centerline"]
        ).astype(np.int32)

        cv2.polylines(
            output,
            [pts],
            False,
            color,
            4,
            cv2.LINE_AA
        )

        middle = len(pts) // 2

        x = int(
            pts[middle][0]
        )

        y = int(
            pts[middle][1]
        )

        label = (
            f"C{i+1} "
            f"{wire['confidence']:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(
        str(path),
        output
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path,
    model
):

    print()
    print("=" * 60)
    print(
        f"IMAGE: {image_path.name}"
    )
    print("=" * 60)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return {
            "image": image_path.name,
            "masks": 0,
            "conductors": 0,
            "pairs": 0
        }

    height, width = image.shape[:2]

    result = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF,
        retina_masks=True,
        verbose=False
    )[0]

    if result.masks is None:

        print("No masks detected.")

        return {
            "image": image_path.name,
            "masks": 0,
            "conductors": 0,
            "pairs": 0
        }

    polygons = result.masks.xy

    classes = (
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
        f"Raw masks: {len(polygons)}"
    )

    candidates = []

    for i, polygon in enumerate(
        polygons
    ):

        class_name = model.names[
            classes[i]
        ]

        if class_name.lower() != "conductor":
            continue

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        polygon = np.round(
            polygon
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        centerline = extract_centerline(
            mask
        )

        if centerline is None:
            continue

        candidates.append({

            "confidence":
                float(confidences[i]),

            "mask":
                mask,

            "centerline":
                centerline

        })

    print(
        f"Conductor candidates: "
        f"{len(candidates)}"
    )

    conductors = remove_duplicates(
        candidates
    )

    print(
        f"Final conductors: "
        f"{len(conductors)}"
    )

    # --------------------------------------------------------
    # SAVE CENTERLINE IMAGE
    # --------------------------------------------------------

    geometry_image = (
        OUTPUT_DIR /
        f"{image_path.stem}_geometry.jpg"
    )

    draw_result(
        image,
        conductors,
        geometry_image
    )

    # --------------------------------------------------------
    # SEPARATIONS
    # --------------------------------------------------------

    pair_rows = []

    for i in range(
        len(conductors)
    ):

        for j in range(
            i + 1,
            len(conductors)
        ):

            result_sep = (
                calculate_separation(
                    conductors[i],
                    conductors[j]
                )
            )

            if result_sep is None:
                continue

            x, separation = result_sep

            mean_sep = float(
                np.mean(separation)
            )

            median_sep = float(
                np.median(separation)
            )

            min_sep = float(
                np.min(separation)
            )

            max_sep = float(
                np.max(separation)
            )

            print(
                f"C{i+1}-C{j+1}: "
                f"mean={mean_sep:.2f}px | "
                f"median={median_sep:.2f}px | "
                f"min={min_sep:.2f}px | "
                f"max={max_sep:.2f}px"
            )

            pair_rows.append({

                "image":
                    image_path.name,

                "wire_1":
                    i + 1,

                "wire_2":
                    j + 1,

                "mean_px":
                    mean_sep,

                "median_px":
                    median_sep,

                "min_px":
                    min_sep,

                "max_px":
                    max_sep

            })

            # CSV
            csv_path = (
                OUTPUT_DIR /
                f"{image_path.stem}_"
                f"C{i+1}_C{j+1}.csv"
            )

            pd.DataFrame({
                "x_px": x,
                "separation_px": separation
            }).to_csv(
                csv_path,
                index=False
            )

            # Graph
            graph_path = (
                OUTPUT_DIR /
                f"{image_path.stem}_"
                f"C{i+1}_C{j+1}.png"
            )

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(
                x,
                separation
            )

            plt.xlabel(
                "Image X Position (px)"
            )

            plt.ylabel(
                "Conductor Separation (px)"
            )

            plt.title(
                f"{image_path.name} "
                f"C{i+1} - C{j+1}"
            )

            plt.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                graph_path,
                dpi=150
            )

            plt.close()

    return {
        "image":
            image_path.name,

        "masks":
            len(polygons),

        "conductors":
            len(conductors),

        "pairs":
            len(pair_rows)
    }


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 65)
print("FINAL WIRE GEOMETRY V3")
print("=" * 65)

print(
    f"Image size : {IMGSZ}"
)

print(
    f"Confidence : {CONF}"
)

print(
    f"Output     : {OUTPUT_DIR}"
)

model = YOLO(
    MODEL_PATH
)

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

if not images:

    raise FileNotFoundError(
        f"No JPG images found in {IMAGE_DIR}"
    )

all_results = []

for i, image_path in enumerate(
    images,
    1
):

    print(
        f"\n[{i}/{len(images)}]"
    )

    result = process_image(
        image_path,
        model
    )

    all_results.append(
        result
    )


# ============================================================
# MASTER CSV
# ============================================================

summary = pd.DataFrame(
    all_results
)

summary_path = (
    OUTPUT_DIR /
    "FINAL_GEOMETRY_SUMMARY.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


print()
print("=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)

print()
print(
    "Results:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Master CSV:"
)

print(
    summary_path
)
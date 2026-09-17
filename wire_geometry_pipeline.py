from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# UNIFIED POWERLINE WIRE GEOMETRY PIPELINE
# ============================================================

MODEL_PATH = r"D:\Powerline_Drone_AI\training_results\wire_segmentation_rtx2050\weights\best.pt"

IMAGE_PATH = r"D:\Powerline_Drone_AI\test_sample\006836.jpg"

OUTPUT_DIR = Path(
    r"D:\Powerline_Drone_AI\wire_geometry_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25

NUM_SAMPLES = 400

SMOOTH_WINDOW = 21

MIN_CENTERLINE_POINTS = 30


# ============================================================
# FUNCTIONS
# ============================================================

def smooth_signal(values, window):

    window = min(
        window,
        len(values)
    )

    if window < 5:
        return values.copy()

    if window % 2 == 0:
        window -= 1

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


def extract_centerline(mask):

    """
    Convert segmentation mask to skeleton.
    """

    skeleton = skeletonize(
        mask > 0
    )

    skeleton_uint8 = (
        skeleton.astype(np.uint8)
    )

    # Find connected components
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


def prepare_curve(x, y):

    """
    Convert skeleton pixels into a smooth
    centerline represented as y(x).
    """

    # Sort by X
    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # Remove duplicate X values
    unique_x = np.unique(x)

    clean_x = []
    clean_y = []

    for value in unique_x:

        ys = y[
            x == value
        ]

        clean_x.append(
            value
        )

        clean_y.append(
            np.median(ys)
        )

    clean_x = np.asarray(
        clean_x,
        dtype=float
    )

    clean_y = np.asarray(
        clean_y,
        dtype=float
    )

    if len(clean_x) < MIN_CENTERLINE_POINTS:

        return None

    # Regular sampling
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

    # Smooth centerline
    sample_y = smooth_signal(
        sample_y,
        SMOOTH_WINDOW
    )

    return sample_x, sample_y


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("UNIFIED POWERLINE WIRE GEOMETRY PIPELINE")
print("=" * 60)
print()

print("Model:")
print(MODEL_PATH)

print()

print("Image:")
print(IMAGE_PATH)

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
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise RuntimeError(
        "Could not read image."
    )

height, width = image.shape[:2]

print(
    f"Image size: {width} x {height}"
)

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("Model loaded.")
print()


# ============================================================
# YOLO SEGMENTATION
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


if result.masks is None:

    raise RuntimeError(
        "No wire masks detected."
    )


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
    f"Masks detected: {len(polygons)}"
)

print()


# ============================================================
# OUTPUT CENTERLINE IMAGE
# ============================================================

centerline_image = image.copy()

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


# ============================================================
# STORE WIRES
# ============================================================

wire_data = {}

wire_id = 0


# ============================================================
# PROCESS MASKS
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
    # Create mask in ORIGINAL image coordinates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Small cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Centerline
    # --------------------------------------------------------

    skeleton = extract_centerline(
        mask
    )

    y_pixels, x_pixels = np.where(
        skeleton
    )

    if len(x_pixels) < MIN_CENTERLINE_POINTS:

        print(
            f"Skipping prediction {i + 1}: "
            "centerline too short."
        )

        continue

    curve = prepare_curve(
        x_pixels,
        y_pixels
    )

    if curve is None:

        print(
            f"Skipping prediction {i + 1}: "
            "unable to create curve."
        )

        continue

    curve_x, curve_y = curve

    wire_id += 1

    wire_data[wire_id] = {

        "class": class_name,

        "confidence": confidence,

        "x": curve_x,

        "y": curve_y,

        "mask": mask

    }

    color = colors[
        (wire_id - 1) %
        len(colors)
    ]

    # --------------------------------------------------------
    # Draw centerline
    # --------------------------------------------------------

    for x, y in zip(
        curve_x,
        curve_y
    ):

        cv2.circle(

            centerline_image,

            (
                int(x),
                int(y)
            ),

            2,

            color,

            -1
        )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    middle = len(
        curve_x
    ) // 2

    cv2.putText(

        centerline_image,

        f"W{wire_id} {class_name}",

        (
            int(curve_x[middle]),
            int(curve_y[middle])
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        color,

        2,

        cv2.LINE_AA
    )

    print(

        f"W{wire_id}: "
        f"{class_name} | "
        f"confidence={confidence:.2f} | "
        f"points={len(curve_x)}"
    )


# ============================================================
# SAVE CENTERLINE IMAGE
# ============================================================

centerline_output = (
    OUTPUT_DIR /
    "006836_centerlines.jpg"
)

cv2.imwrite(

    str(centerline_output),

    centerline_image
)

print()

print(
    f"Centerline image saved:"
)

print(
    centerline_output
)

print()


# ============================================================
# IDENTIFY CONDUCTORS
# ============================================================

conductor_ids = [

    wid

    for wid, data
    in wire_data.items()

    if data["class"].lower()
    == "conductor"
]


print(
    f"Conductors found: "
    f"{len(conductor_ids)}"
)

print()


if len(conductor_ids) < 2:

    raise RuntimeError(
        "At least two conductors are "
        "required for separation."
    )


# ============================================================
# SEPARATION CALCULATION
# ============================================================

all_results = []

profile_records = []

print(
    "Calculating conductor separation..."
)

print()


for a in range(
    len(conductor_ids)
):

    for b in range(
        a + 1,
        len(conductor_ids)
    ):

        id_a = conductor_ids[a]

        id_b = conductor_ids[b]

        wire_a = wire_data[id_a]

        wire_b = wire_data[id_b]

        xa = wire_a["x"]

        ya = wire_a["y"]

        xb = wire_b["x"]

        yb = wire_b["y"]

        # ----------------------------------------------------
        # Common region
        # ----------------------------------------------------

        start_x = max(
            xa.min(),
            xb.min()
        )

        end_x = min(
            xa.max(),
            xb.max()
        )

        if end_x <= start_x:

            print(
                f"W{id_a}-W{id_b}: "
                "no common region"
            )

            continue

        # ----------------------------------------------------
        # Common sampling positions
        # ----------------------------------------------------

        x = np.linspace(

            start_x,

            end_x,

            NUM_SAMPLES
        )

        ya_sample = np.interp(
            x,
            xa,
            ya
        )

        yb_sample = np.interp(
            x,
            xb,
            yb
        )

        # ----------------------------------------------------
        # Local tangent of conductor A
        # ----------------------------------------------------

        dydx = np.gradient(
            ya_sample,
            x
        )

        # ----------------------------------------------------
        # Unit normal
        #
        # Tangent = (1, dydx)
        #
        # Normal = (-dydx, 1)
        # ----------------------------------------------------

        nx = -dydx

        ny = np.ones_like(
            dydx
        )

        length = np.sqrt(
            nx ** 2 +
            ny ** 2
        )

        nx /= length

        ny /= length

        # ----------------------------------------------------
        # Vector A → B
        # ----------------------------------------------------

        dx = np.zeros_like(
            x
        )

        dy = (
            yb_sample -
            ya_sample
        )

        # ----------------------------------------------------
        # PERPENDICULAR DISTANCE
        # ----------------------------------------------------

        separation = np.abs(

            dx * nx +

            dy * ny

        )

        # ----------------------------------------------------
        # Smooth separation
        # ----------------------------------------------------

        smooth_separation = (
            smooth_signal(
                separation,
                SMOOTH_WINDOW
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mean_sep = float(
            np.mean(
                smooth_separation
            )
        )

        median_sep = float(
            np.median(
                smooth_separation
            )
        )

        min_index = int(
            np.argmin(
                smooth_separation
            )
        )

        max_index = int(
            np.argmax(
                smooth_separation
            )
        )

        min_sep = float(
            smooth_separation[
                min_index
            ]
        )

        max_sep = float(
            smooth_separation[
                max_index
            ]
        )

        std_sep = float(
            np.std(
                smooth_separation
            )
        )

        min_x = float(
            x[min_index]
        )

        max_x = float(
            x[max_index]
        )

        # ----------------------------------------------------
        # Save pair summary
        # ----------------------------------------------------

        all_results.append({

            "wire_1": id_a,

            "wire_2": id_b,

            "mean_separation_px":
                mean_sep,

            "median_separation_px":
                median_sep,

            "minimum_separation_px":
                min_sep,

            "maximum_separation_px":
                max_sep,

            "std_separation_px":
                std_sep

        })

        # ----------------------------------------------------
        # Save profile
        # ----------------------------------------------------

        for i in range(
            len(x)
        ):

            profile_records.append({

                "wire_1": id_a,

                "wire_2": id_b,

                "x_position_px":
                    x[i],

                "wire_1_y_px":
                    ya_sample[i],

                "wire_2_y_px":
                    yb_sample[i],

                "separation_px":
                    separation[i],

                "smoothed_separation_px":
                    smooth_separation[i]

            })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(

            f"W{id_a} - W{id_b}: "

            f"mean={mean_sep:.2f}px | "

            f"median={median_sep:.2f}px | "

            f"min={min_sep:.2f}px | "

            f"max={max_sep:.2f}px"

        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = (
    OUTPUT_DIR /
    "006836_geometry_summary.csv"
)

pd.DataFrame(
    all_results
).to_csv(
    summary_csv,
    index=False
)


# ============================================================
# SAVE PROFILE CSV
# ============================================================

profile_csv = (
    OUTPUT_DIR /
    "006836_geometry_profile.csv"
)

pd.DataFrame(
    profile_records
).to_csv(
    profile_csv,
    index=False
)


# ============================================================
# GRAPH
# ============================================================

if len(profile_records) > 0:

    profile_df = pd.DataFrame(
        profile_records
    )

    # Current image has two conductors
    pair = profile_df.iloc[0]

    pair_data = profile_df[
        (profile_df["wire_1"]
         == pair["wire_1"])
        &
        (profile_df["wire_2"]
         == pair["wire_2"])
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(

        pair_data[
            "x_position_px"
        ],

        pair_data[
            "separation_px"
        ],

        linewidth=1,

        label="Raw separation"
    )

    plt.plot(

        pair_data[
            "x_position_px"
        ],

        pair_data[
            "smoothed_separation_px"
        ],

        linewidth=2,

        label="Smoothed separation"
    )

    min_row = pair_data.loc[
        pair_data[
            "smoothed_separation_px"
        ].idxmin()
    ]

    max_row = pair_data.loc[
        pair_data[
            "smoothed_separation_px"
        ].idxmax()
    ]

    plt.scatter(

        min_row[
            "x_position_px"
        ],

        min_row[
            "smoothed_separation_px"
        ],

        s=60,

        label=(
            f"Minimum "
            f"{min_row['smoothed_separation_px']:.2f}px"
        )
    )

    plt.scatter(

        max_row[
            "x_position_px"
        ],

        max_row[
            "smoothed_separation_px"
        ],

        s=60,

        label=(
            f"Maximum "
            f"{max_row['smoothed_separation_px']:.2f}px"
        )
    )

    plt.xlabel(
        "Image X Position (pixels)"
    )

    plt.ylabel(
        "Conductor Separation (pixels)"
    )

    plt.title(
        "Conductor Separation Profile"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    graph_path = (
        OUTPUT_DIR /
        "006836_separation_profile.png"
    )

    plt.savefig(
        graph_path,
        dpi=200
    )

    plt.close()


# ============================================================
# ANNOTATED MINIMUM SEPARATION
# ============================================================

if len(profile_records) > 0:

    profile_df = pd.DataFrame(
        profile_records
    )

    pair = profile_df.iloc[0]

    pair_data = profile_df[
        (profile_df["wire_1"]
         == pair["wire_1"])
        &
        (profile_df["wire_2"]
         == pair["wire_2"])
    ]

    min_row = pair_data.loc[
        pair_data[
            "smoothed_separation_px"
        ].idxmin()
    ]

    min_x = int(
        min_row[
            "x_position_px"
        ]
    )

    min_y1 = int(
        min_row[
            "wire_1_y_px"
        ]
    )

    min_y2 = int(
        min_row[
            "wire_2_y_px"
        ]
    )

    annotated = image.copy()

    # Minimum separation points

    cv2.circle(
        annotated,
        (min_x, min_y1),
        10,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        annotated,
        (min_x, min_y2),
        10,
        (0, 0, 255),
        -1
    )

    # Connecting measurement line

    cv2.line(

        annotated,

        (min_x, min_y1),

        (min_x, min_y2),

        (0, 0, 255),

        4
    )

    label = (
        f"MIN SEPARATION = "
        f"{min_row['smoothed_separation_px']:.2f} px"
    )

    cv2.putText(

        annotated,

        label,

        (
            min_x,
            min(min_y1, min_y2) - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 0, 255),

        2,

        cv2.LINE_AA
    )

    annotated_path = (
        OUTPUT_DIR /
        "006836_min_separation.jpg"
    )

    cv2.imwrite(

        str(annotated_path),

        annotated
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("GEOMETRY PIPELINE COMPLETE")
print("=" * 60)
print()

for result in all_results:

    print(

        f"W{result['wire_1']}-"
        f"W{result['wire_2']}"

    )

    print(

        f"  Mean   : "
        f"{result['mean_separation_px']:.2f} px"

    )

    print(

        f"  Median : "
        f"{result['median_separation_px']:.2f} px"

    )

    print(

        f"  Minimum: "
        f"{result['minimum_separation_px']:.2f} px"

    )

    print(

        f"  Maximum: "
        f"{result['maximum_separation_px']:.2f} px"

    )

    print(

        f"  Std    : "
        f"{result['std_separation_px']:.2f} px"

    )

    print()


print("Outputs:")
print()

print(
    f"Centerlines:"
)

print(
    centerline_output
)

print()

print(
    f"Summary:"
)

print(
    summary_csv
)

print()

print(
    f"Profile:"
)

print(
    profile_csv
)

print()

print(
    f"Graph:"
)

print(
    OUTPUT_DIR /
    "006836_separation_profile.png"
)

print()

print(
    f"Minimum separation image:"
)

print(
    OUTPUT_DIR /
    "006836_min_separation.jpg"
)

print()

print(
    "All distances are IMAGE-SPACE PIXELS."
)

print(
    "No physical-distance conversion "
    "has been applied."
)

print()
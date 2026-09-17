import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# ============================================================
# WIRE SEPARATION CALCULATION
# ============================================================

CSV_PATH = r"D:\Powerline_Drone_AI\centerline_results\006836_centerlines_corrected.csv"

OUTPUT_DIR = Path(r"D:\Powerline_Drone_AI\centerline_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "006836_wire_separation.csv"

# Number of points used for comparison along the common span
NUM_SAMPLES = 300

# Smoothing window
SMOOTH_WINDOW = 15


# ============================================================
# LOAD CENTERLINE DATA
# ============================================================

print()
print("==========================================")
print("WIRE SEPARATION CALCULATION")
print("==========================================")
print()

if not Path(CSV_PATH).exists():
    raise FileNotFoundError(
        f"Centerline CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("Centerline CSV loaded.")
print()

# ============================================================
# SHOW DETECTED WIRES
# ============================================================

print("Detected wires:")

for wire_id in sorted(df["wire_id"].unique()):

    wire_data = df[df["wire_id"] == wire_id]

    print(
        f"  W{wire_id}: "
        f"{wire_data['class'].iloc[0]} | "
        f"confidence={wire_data['confidence'].iloc[0]:.2f} | "
        f"points={len(wire_data)}"
    )

print()

# ============================================================
# KEEP ONLY CONDUCTORS
# ============================================================

conductors = df[
    df["class"].str.lower() == "conductor"
].copy()

if conductors.empty:

    raise RuntimeError(
        "No conductor centerlines were found."
    )

conductor_ids = sorted(
    conductors["wire_id"].unique()
)

print(
    f"Conductor wires found: "
    f"{len(conductor_ids)}"
)

print()

if len(conductor_ids) < 2:

    raise RuntimeError(
        "At least two conductors are required "
        "to calculate separation."
    )


# ============================================================
# PREPARE EACH CENTERLINE
# ============================================================

def prepare_centerline(wire_data):

    """
    Convert unordered skeleton pixels into
    a smooth y(x) representation.

    This works well for the current dataset
    because the conductors generally progress
    across the image horizontally.
    """

    wire_data = wire_data.sort_values("x")

    x = wire_data["x"].to_numpy(dtype=float)
    y = wire_data["y"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Combine multiple pixels having the same x coordinate
    # --------------------------------------------------------

    unique_x = np.unique(x)

    clean_x = []
    clean_y = []

    for value in unique_x:

        y_values = y[x == value]

        clean_x.append(value)
        clean_y.append(np.median(y_values))

    clean_x = np.array(clean_x)
    clean_y = np.array(clean_y)

    # --------------------------------------------------------
    # Remove duplicate x values
    # --------------------------------------------------------

    if len(clean_x) < 5:

        raise RuntimeError(
            "Centerline contains too few points."
        )

    # --------------------------------------------------------
    # Interpolate onto regular x positions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Smooth y coordinates
    # --------------------------------------------------------

    window = min(
        SMOOTH_WINDOW,
        len(sample_y)
    )

    if window % 2 == 0:
        window -= 1

    if window >= 5:

        kernel = np.ones(window) / window

        padded = np.pad(
            sample_y,
            window // 2,
            mode="edge"
        )

        sample_y = np.convolve(
            padded,
            kernel,
            mode="valid"
        )

    return sample_x, sample_y


# ============================================================
# PREPARE ALL CONDUCTORS
# ============================================================

centerlines = {}

for wire_id in conductor_ids:

    wire_data = conductors[
        conductors["wire_id"] == wire_id
    ]

    x, y = prepare_centerline(
        wire_data
    )

    centerlines[wire_id] = {
        "x": x,
        "y": y
    }


# ============================================================
# CALCULATE PERPENDICULAR SEPARATION
# ============================================================

results = []

print("Calculating conductor separation...")
print()

for index_a in range(
    len(conductor_ids)
):

    wire_a = conductor_ids[index_a]

    for index_b in range(
        index_a + 1,
        len(conductor_ids)
    ):

        wire_b = conductor_ids[index_b]

        x_a = centerlines[wire_a]["x"]
        y_a = centerlines[wire_a]["y"]

        x_b = centerlines[wire_b]["x"]
        y_b = centerlines[wire_b]["y"]

        # ----------------------------------------------------
        # Find common x region
        # ----------------------------------------------------

        start_x = max(
            x_a.min(),
            x_b.min()
        )

        end_x = min(
            x_a.max(),
            x_b.max()
        )

        if end_x <= start_x:

            print(
                f"W{wire_a}-W{wire_b}: "
                "No common image region."
            )

            continue

        # ----------------------------------------------------
        # Sample common region
        # ----------------------------------------------------

        sample_x = np.linspace(
            start_x,
            end_x,
            NUM_SAMPLES
        )

        y_a_sample = np.interp(
            sample_x,
            x_a,
            y_a
        )

        y_b_sample = np.interp(
            sample_x,
            x_b,
            y_b
        )

        # ----------------------------------------------------
        # Estimate tangent of wire A
        # ----------------------------------------------------

        dy_dx = np.gradient(
            y_a_sample,
            sample_x
        )

        # Tangent vector:
        #
        # T = (1, dy/dx)
        #
        # Normal vector:
        #
        # N = (-dy/dx, 1)
        # ----------------------------------------------------

        normal_x = -dy_dx

        normal_y = np.ones_like(
            dy_dx
        )

        normal_length = np.sqrt(
            normal_x ** 2 +
            normal_y ** 2
        )

        normal_x /= normal_length
        normal_y /= normal_length

        # ----------------------------------------------------
        # Vector from wire A to wire B
        # ----------------------------------------------------

        delta_x = np.zeros_like(
            sample_x
        )

        delta_y = (
            y_b_sample -
            y_a_sample
        )

        # ----------------------------------------------------
        # Perpendicular distance
        # ----------------------------------------------------

        perpendicular_distance = np.abs(
            delta_x * normal_x +
            delta_y * normal_y
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mean_distance = float(
            np.mean(
                perpendicular_distance
            )
        )

        median_distance = float(
            np.median(
                perpendicular_distance
            )
        )

        min_distance = float(
            np.min(
                perpendicular_distance
            )
        )

        max_distance = float(
            np.max(
                perpendicular_distance
            )
        )

        std_distance = float(
            np.std(
                perpendicular_distance
            )
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "wire_1": wire_a,

            "wire_2": wire_b,

            "samples": len(
                perpendicular_distance
            ),

            "mean_separation_px":
                mean_distance,

            "median_separation_px":
                median_distance,

            "minimum_separation_px":
                min_distance,

            "maximum_separation_px":
                max_distance,

            "std_separation_px":
                std_distance
        })

        print(
            f"W{wire_a} - W{wire_b}: "
            f"mean={mean_distance:.2f}px | "
            f"median={median_distance:.2f}px | "
            f"min={min_distance:.2f}px | "
            f"max={max_distance:.2f}px"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

if not results:

    raise RuntimeError(
        "No wire separation results were calculated."
    )

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    SUMMARY_PATH,
    index=False
)

print()
print("==========================================")
print("SEPARATION CALCULATION COMPLETE")
print("==========================================")
print()

print(
    f"Results saved to:"
)

print(
    SUMMARY_PATH
)

print()

print(
    "IMPORTANT:"
)

print(
    "These measurements are image-space "
    "separations in pixels."
)

print(
    "They are NOT physical distances "
    "in metres or centimetres."
)

print()
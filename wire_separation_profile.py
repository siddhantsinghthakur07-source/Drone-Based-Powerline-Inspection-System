import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# WIRE SEPARATION PROFILE
# ============================================================

CSV_PATH = r"D:\Powerline_Drone_AI\centerline_results\006836_centerlines_corrected.csv"

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

# We specifically want conductor-to-conductor separation
WIRE_A = 2
WIRE_B = 3

# Number of locations sampled along the common span
NUM_SAMPLES = 300

# Smoothing applied to separation profile
SMOOTH_WINDOW = 15


# ============================================================
# START
# ============================================================

print()
print("==========================================")
print("WIRE SEPARATION PROFILE")
print("==========================================")
print()

# ============================================================
# CHECK FILES
# ============================================================

if not Path(CSV_PATH).exists():

    raise FileNotFoundError(
        f"Centerline CSV not found:\n{CSV_PATH}"
    )

if not Path(IMAGE_PATH).exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Centerline data loaded.")
print()

# ============================================================
# GET TWO CONDUCTORS
# ============================================================

wire_a_data = df[
    df["wire_id"] == WIRE_A
].copy()

wire_b_data = df[
    df["wire_id"] == WIRE_B
].copy()

if wire_a_data.empty:

    raise RuntimeError(
        f"Wire {WIRE_A} was not found."
    )

if wire_b_data.empty:

    raise RuntimeError(
        f"Wire {WIRE_B} was not found."
    )

# Confirm they are conductors

class_a = wire_a_data["class"].iloc[0]
class_b = wire_b_data["class"].iloc[0]

if class_a.lower() != "conductor":

    raise RuntimeError(
        f"W{WIRE_A} is {class_a}, not conductor."
    )

if class_b.lower() != "conductor":

    raise RuntimeError(
        f"W{WIRE_B} is {class_b}, not conductor."
    )


print(
    f"Comparing W{WIRE_A} "
    f"with W{WIRE_B}"
)

print()


# ============================================================
# PREPARE CENTERLINE
# ============================================================

def prepare_centerline(wire_data):

    wire_data = wire_data.sort_values(
        "x"
    )

    x = wire_data[
        "x"
    ].to_numpy(
        dtype=float
    )

    y = wire_data[
        "y"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Remove duplicate x coordinates
    # --------------------------------------------------------

    unique_x = np.unique(x)

    clean_x = []
    clean_y = []

    for value in unique_x:

        y_values = y[
            x == value
        ]

        clean_x.append(
            value
        )

        clean_y.append(
            np.median(
                y_values
            )
        )

    clean_x = np.array(
        clean_x
    )

    clean_y = np.array(
        clean_y
    )

    if len(clean_x) < 5:

        raise RuntimeError(
            "Not enough centerline points."
        )

    return clean_x, clean_y


x_a, y_a = prepare_centerline(
    wire_a_data
)

x_b, y_b = prepare_centerline(
    wire_b_data
)


# ============================================================
# FIND COMMON IMAGE REGION
# ============================================================

start_x = max(
    x_a.min(),
    x_b.min()
)

end_x = min(
    x_a.max(),
    x_b.max()
)

if end_x <= start_x:

    raise RuntimeError(
        "The two conductors have no "
        "common image region."
    )


# ============================================================
# SAMPLE COMMON REGION
# ============================================================

sample_x = np.linspace(
    start_x,
    end_x,
    NUM_SAMPLES
)

sample_y_a = np.interp(
    sample_x,
    x_a,
    y_a
)

sample_y_b = np.interp(
    sample_x,
    x_b,
    y_b
)


# ============================================================
# CALCULATE LOCAL TANGENT OF WIRE A
# ============================================================

dy_dx = np.gradient(
    sample_y_a,
    sample_x
)


# ============================================================
# LOCAL NORMAL
# ============================================================

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


# ============================================================
# DISTANCE BETWEEN WIRES
# ============================================================

delta_x = (
    sample_x -
    sample_x
)

delta_y = (
    sample_y_b -
    sample_y_a
)

separation = np.abs(
    delta_x * normal_x +
    delta_y * normal_y
)


# ============================================================
# SMOOTH SEPARATION
# ============================================================

window = min(
    SMOOTH_WINDOW,
    len(separation)
)

if window % 2 == 0:

    window -= 1

if window >= 5:

    kernel = (
        np.ones(window) /
        window
    )

    padded = np.pad(
        separation,
        window // 2,
        mode="edge"
    )

    smooth_separation = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

else:

    smooth_separation = separation.copy()


# ============================================================
# STATISTICS
# ============================================================

mean_sep = float(
    np.mean(
        separation
    )
)

median_sep = float(
    np.median(
        separation
    )
)

min_index = int(
    np.argmin(
        separation
    )
)

max_index = int(
    np.argmax(
        separation
    )
)

min_sep = float(
    separation[min_index]
)

max_sep = float(
    separation[max_index]
)

min_x = float(
    sample_x[min_index]
)

min_y_a = float(
    sample_y_a[min_index]
)

min_y_b = float(
    sample_y_b[min_index]
)

max_x = float(
    sample_x[max_index]
)

max_y_a = float(
    sample_y_a[max_index]
)

max_y_b = float(
    sample_y_b[max_index]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("==========================================")
print("SEPARATION RESULTS")
print("==========================================")
print()

print(
    f"Mean separation   : "
    f"{mean_sep:.2f} px"
)

print(
    f"Median separation : "
    f"{median_sep:.2f} px"
)

print(
    f"Minimum separation: "
    f"{min_sep:.2f} px"
)

print(
    f"Maximum separation: "
    f"{max_sep:.2f} px"
)

print()

print(
    f"Minimum location: "
    f"x={min_x:.1f}"
)

print(
    f"Maximum location: "
    f"x={max_x:.1f}"
)

print()


# ============================================================
# SAVE PROFILE CSV
# ============================================================

profile_csv = (
    OUTPUT_DIR /
    "006836_separation_profile.csv"
)

profile_df = pd.DataFrame({

    "x_position_px":
        sample_x,

    "wire_2_y_px":
        sample_y_a,

    "wire_3_y_px":
        sample_y_b,

    "separation_px":
        separation,

    "smoothed_separation_px":
        smooth_separation

})

profile_df.to_csv(
    profile_csv,
    index=False
)


# ============================================================
# CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    sample_x,
    separation,
    linewidth=1,
    label="Raw separation"
)

plt.plot(
    sample_x,
    smooth_separation,
    linewidth=2,
    label="Smoothed separation"
)

plt.scatter(
    [min_x],
    [min_sep],
    s=60,
    label=f"Minimum = {min_sep:.2f} px"
)

plt.scatter(
    [max_x],
    [max_sep],
    s=60,
    label=f"Maximum = {max_sep:.2f} px"
)

plt.xlabel(
    "Image X Position (pixels)"
)

plt.ylabel(
    "Conductor Separation (pixels)"
)

plt.title(
    f"Wire Separation Profile: "
    f"W{WIRE_A} - W{WIRE_B}"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


profile_plot = (
    OUTPUT_DIR /
    "006836_separation_profile.png"
)

plt.savefig(
    profile_plot,
    dpi=200
)

plt.close()


# ============================================================
# CREATE ANNOTATED IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise RuntimeError(
        "Could not read image."
    )

# Minimum separation point

min_point_a = (
    int(min_x),
    int(min_y_a)
)

min_point_b = (
    int(min_x),
    int(min_y_b)
)

# Draw points

cv2.circle(
    image,
    min_point_a,
    10,
    (0, 0, 255),
    -1
)

cv2.circle(
    image,
    min_point_b,
    10,
    (0, 0, 255),
    -1
)

# Draw line showing minimum separation

cv2.line(
    image,
    min_point_a,
    min_point_b,
    (0, 0, 255),
    4
)

# Label

label = (
    f"MIN SEPARATION: "
    f"{min_sep:.2f} px"
)

cv2.putText(
    image,
    label,
    (
        min_point_a[0],
        min_point_a[1] - 20
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================

annotated_image = (
    OUTPUT_DIR /
    "006836_min_separation.jpg"
)

cv2.imwrite(
    str(annotated_image),
    image
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("==========================================")
print("SEPARATION PROFILE COMPLETE")
print("==========================================")
print()

print(
    f"Profile CSV:"
)

print(
    profile_csv
)

print()

print(
    f"Profile graph:"
)

print(
    profile_plot
)

print()

print(
    f"Annotated image:"
)

print(
    annotated_image
)

print()

print(
    "All measurements are in PIXELS."
)

print(
    "No physical-distance conversion has "
    "been performed."
)

print()
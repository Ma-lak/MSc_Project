
import numpy as np
import tifffile as tiff
import cv2

from scipy.ndimage import gaussian_filter
from scipy.fft import dctn
from scipy.ndimage import convolve

from sklearn.metrics import mean_absolute_error

# ==============================
# SETTINGS
# ==============================
fish1 = "data/fish1.tif"
fish2 = "data/fish2.tif"
fish3 = "data/fish3.tif"
fish4 = "data/fish4.tif"
fish5 = "data/fish5.tif"
fish6_z1 = "data/fish6_zstack1.tif"
fish6_z2 = "data/fish6_zstack2.tif"
fish7_z1 = "data/fish7_zstack1.tif"
fish7_z2 = "data/fish7_zstack2.tif"
fish8 = "data/fish8.tif"
s3 = "data/uclaminiscopev4-stack_0.tif"
s2 = "data/uclaminiscopev4-stack_1_40fps.tif"


z_spacing = 16      
offset = 0           

stack = tiff.imread(fish6_z1)

print("Stack shape:", stack.shape)


# ============================================================
# GROUND TRUTH FOCAL DISTANCES
# ============================================================

n_frames = stack.shape[0]

real_focus = (
    (np.arange(n_frames) - n_frames // 2)
    * z_spacing
    + offset
)


# ============================================================
# 1. TENENGRAD VARIANCE
# ============================================================

def tenengrad(image, sigma=1):

    image = gaussian_filter(
        image.astype(np.float32),
        sigma=sigma
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = gx**2 + gy**2

    return np.mean(gradient)


# ============================================================
# 2. LAPLACIAN VARIANCE
# ============================================================

def laplacian_variance(image, sigma=1):

    image = gaussian_filter(
        image.astype(np.float32),
        sigma=sigma
    )

    lap = cv2.Laplacian(
        image,
        cv2.CV_32F
    )

    return np.var(lap)


# ============================================================
# 3. SHANNON ENTROPY
# ============================================================

def shannon_entropy(image):

    image = image.astype(np.float32)

    # Convert intensities into histogram probabilities
    hist, _ = np.histogram(
        image,
        bins=256,
        density=True
    )

    hist = hist[hist > 0]

    return -np.sum(
        hist * np.log2(hist)
    )






# ============================================================
# 6. BRENNER'S MEASURE
# ============================================================

def brenner(image):

    image = image.astype(np.float32)

    # Difference between pixels separated by 2
    diff = (
        image[:, 2:] -
        image[:, :-2]
    )

    return np.mean(
        diff**2
    )


# ============================================================
# 7. STEERABLE FILTERS
# ============================================================

def steerable_filter_score(image):

    image = image.astype(np.float32)

    # Gaussian derivative filters
    # Different orientations

    sigma = 1.5

    # First derivatives
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=5
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=5
    )

    # Second derivatives
    gxx = cv2.Sobel(
        image,
        cv2.CV_32F,
        2,
        0,
        ksize=5
    )

    gyy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        2,
        ksize=5
    )

    gxy = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        1,
        ksize=5
    )

    # Steerable second-order response
    angles = np.linspace(
        0,
        np.pi,
        18,
        endpoint=False
    )

    responses = []

    for theta in angles:

        c = np.cos(theta)
        s = np.sin(theta)

        response = (
            c**2 * gxx
            + 2*c*s*gxy
            + s**2 * gyy
        )

        responses.append(
            response**2
        )

    responses = np.array(
        responses
    )

    # Maximum response over orientations
    max_response = np.max(
        responses,
        axis=0
    )

    return np.mean(
        max_response
    )


# ============================================================
# METHODS
# ============================================================

methods = {

    "Tenengrad": tenengrad,

    "Laplacian variance":
        laplacian_variance,

    "Shannon entropy":
        shannon_entropy,

    "Brenner":
        brenner,

    "Steerable filter":
        steerable_filter_score
}


# ============================================================
# RUN BASELINES
# ============================================================

results = {}

for name, method in methods.items():

    print(
        f"\nRunning {name}..."
    )

    scores = []

    for i, image in enumerate(stack):

        score = method(image)

        scores.append(score)

    scores = np.array(scores)

    # Highest score = predicted best focus
    best_frame = np.argmax(scores)

    # Convert frames to predicted focal distances
    predicted_distances = (
        np.arange(n_frames)
        - best_frame
    ) * z_spacing

    # MAE against ground truth
    mae = mean_absolute_error(
        real_focus,
        predicted_distances
    )

    results[name] = {
        "scores": scores,
        "best_frame": best_frame,
        "predicted_distances":
            predicted_distances,
        "MAE": mae
    }

    print(
        f"Best focus frame: {best_frame}"
    )

    print(
        f"MAE: {mae:.2f} µm"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 50)
print("BASELINE COMPARISON")
print("=" * 50)

for name, result in results.items():

    print(
        f"{name:25s} "
        f"Best frame: {result['best_frame']:3d}   "
        f"MAE: {result['MAE']:.2f} µm"
    )

# ==============================
# COMPARE BASELINE MAE
# ==============================

import matplotlib.pyplot as plt

results["MobileNetV2"] = {
    "MAE": 22.83
}
names = list(results.keys())
mae_values = [results[name]["MAE"] for name in names]


plt.figure(figsize=(10, 6))

bars = plt.bar(names, mae_values, color=plt.cm.tab10(np.arange(len(names))))

plt.ylabel("MAE (µm)", fontsize=14)
plt.xlabel("Method", fontsize=14)
plt.title("Comparison of Autofocus Methods", fontsize=16)

plt.xticks(rotation=35, ha="right", fontsize=11)
plt.yticks(fontsize=11)

# Add values above bars
for bar, value in zip(bars, mae_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    "baseline_results/baseline_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

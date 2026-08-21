import numpy as np
import os
from PIL import Image
from scipy.signal import savgol_filter, medfilt2d
import tifffile as tiff


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


csv_file = "output/file_names_and_distances_combined.csv"
image_folder = "output"

# z values from +100 down to -100
distances = list(range(100, -101, -40))

df = pd.read_csv(csv_file)


# --------------------------------------------------
# EXTRACT PATCH NUMBER FROM FILE NAME
# --------------------------------------------------


# combined_data_patch0_d-100.tif
#                    ^
#                    patch number = 0

df["patch_id"] = (
    df["file_name"]
    .str.extract(r"patch(\d+)")[0]
    .astype(int)
)


# --------------------------------------------------
# SELECT 3 RANDOM PATCHES ONCE
# --------------------------------------------------

# Use z = 0 as the reference slice
reference_z = 0

reference_candidates = df[
    df["distance"] == reference_z
]

# Randomly select 3 patch numbers
# random_state makes the selection reproducible
selected_patch_ids = (
    reference_candidates["patch_id"]
    .sample(
        n=3,
        random_state=42
    )
    .tolist()
)

print("Selected patches:", selected_patch_ids)



# DISPLAY IMAGE


def display_image(img, percentile=99):

    img = img.astype(np.float32)

    # 99th percentile
    vmax = np.percentile(
        img,
        percentile
    )

    vmin = np.percentile(
        img,
        1
    )

    img = np.clip(
        img,
        vmin,
        vmax
    )

    return img, vmin, vmax


# --------------------------------------------------
# PLOT

n_rows = len(distances)
n_cols = 3

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(9, n_rows * 2.5)
)

# Make sure axes is 2D
axes = np.atleast_2d(axes)


for i, d in enumerate(distances):

    for j, patch_id in enumerate(selected_patch_ids):

        # Find the file corresponding to:
        # current z value
        # current patch number

        match = df[
            (df["distance"] == d) &
            (df["patch_id"] == patch_id)
        ]

        if len(match) == 0:

            axes[i, j].axis("off")

            continue


        # Get file info
        row = match.iloc[0]

        path = os.path.join(
            image_folder,
            row["file_name"]
        )


        # Load image
        img = np.array(
            Image.open(path)
        )


        # Adjust display range
        img, vmin, vmax = display_image(img)


        # Display
        axes[i, j].imshow(
            img,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )


        axes[i, j].axis("off")


    # Add z label
    axes[i, 0].set_ylabel(
        f"z = {d} µm",
        fontsize=11,
        rotation=0,
        labelpad=45,
        va="center"
    )



# COLUMN LABELS


for j, patch_id in enumerate(selected_patch_ids):

    axes[0, j].set_title(
        f"Patch {patch_id}",
        fontsize=12
    )



# Z AXIS

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.95,
    bottom=0.03,
    hspace=0.15,
    wspace=0.05
)


# Create separate z-axis
ax_z = fig.add_axes(
    [0.05, 0.03, 0.02, 0.92]
)


# z goes from +100 at top
# to -100 at bottom
ax_z.set_ylim(
    100,
    -100
)


ax_z.set_xlim(
    0,
    1
)


ax_z.set_yticks(
    distances
)


ax_z.set_ylabel(
    "z (µm)",
    fontsize=13
)


# Hide x-axis
ax_z.set_xticks([])


# Only show left spine
ax_z.spines["right"].set_visible(False)
ax_z.spines["top"].set_visible(False)
ax_z.spines["bottom"].set_visible(False)

plt.savefig("combined_data/patches.png", dpi=300, bbox_inches="tight")
plt.show()

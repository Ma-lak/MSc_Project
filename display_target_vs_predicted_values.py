import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import tifffile as tiff


# Paths

s3 = "data/uclaminiscopev4-stack_0.tif"
s2 = "data/uclaminiscopev4-stack_1_40fps.tif"
fish6_z1 = "data/fish6_zstack1.tif"
fish6_z2 = "data/fish6_zstack2.tif"
fish7_z1 = "data/fish7_zstack1.tif"
fish7_z2 = "data/fish7_zstack2.tif"
fish4 = "data/fish4.tif"
fish1 = "data/fish1.tif"
fish2 = "data/fish2.tif"
fish3 = "data/fish3.tif"
fish5 = "data/fish5.tif"
fish8 = "data/fish8.tif"

checkpoint_path = "modelA_s42_2_checkpoint_700.pth"


# Image processing

def process_image(image, sat_prctile=99):
    sat = np.percentile(image, sat_prctile)
    image = sat * np.tanh(image / (sat + 1e-8))
    image = image / np.sqrt(np.sum(image ** 2) + 1e-8)
    return image

 
def select_patches(image, patch_size, num_patches, threshold):
    patches = []

    x_min = image.shape[0] // 4
    x_max = image.shape[0] * 3 // 4
    y_min = image.shape[1] // 4
    y_max = image.shape[1] * 3 // 4

    for _ in range(num_patches):
        while True:
            x = np.random.randint(x_min, x_max - patch_size)
            y = np.random.randint(y_min, y_max - patch_size)

            patch = image[
                x:x + patch_size,
                y:y + patch_size
            ]

            if np.percentile(patch, 99) > threshold:
                patches.append((x, y))
                break

    return patches



# Load model

model = MobileNetV2()

model.features[0][0] = Conv2dNormActivation(
    1,
    32,
    kernel_size=1,
    norm_layer=None,
    activation_layer=nn.ReLU
)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 3)
)

device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model.to(device)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()



# Load TIFF stack

images = tiff.imread(fish6_z2)   # change depending on stack

print("Loaded TIFF stack:", images.shape)


# Parameters

offset = -6 
img_range = range(0, images.shape[0]-1)

patch_size = 224
num_patches = 128 
threshold = 25
distance_between_images = 16


# Run inference

dists = []
real_focus = []

for img in img_range:

    this_real_focus = (
        (img - images.shape[0] // 2)
        * distance_between_images
        + offset
    )

    print("Processing image:", img)

    this_image = images[img, :, :]

    # Calculate real focal distance
    real_focus.append(this_real_focus)

    # Select patches
    selected_patches = select_patches(
        this_image,
        patch_size,
        num_patches,
        threshold
    )

    all_patches = []

    for patch in selected_patches:

        this_patch = this_image[
            patch[0]:patch[0] + patch_size,
            patch[1]:patch[1] + patch_size
        ].astype(np.float32)

        # 2x2 binning
        this_patch = (
            this_patch
            .reshape(
                this_patch.shape[0] // 2,
                2,
                this_patch.shape[1] // 2,
                2
            )
            .sum(axis=(1, 3))
        )

        # Normalise
        this_patch = process_image(
            this_patch,
            sat_prctile=95
        )

        all_patches.append(this_patch)

    # Convert patches to tensor
    all_patches = transforms.ToImage()(
        np.array(all_patches)
    )

    all_patches = (
        all_patches
        .permute(1, 2, 0)
        .unsqueeze(1)
        .to(device)
    )

    # Model prediction
    with torch.no_grad():
        pred = model(all_patches).cpu().numpy()

    # Mean predicted distance
    mean_distance = np.mean(pred[:, 0] * 40)

    # Determine direction/class
    mean_class = np.mean(
        np.argmax(pred[:, 1:], axis=1)
    )

    if mean_class < 0.5:
        mean_distance *= -1

    dists.append(mean_distance)


# Plot graph only

plt.figure(figsize=(10, 6))

plt.plot(
    real_focus,
    label="Real",
    color="black",
    linewidth=3
)

real_focus = np.array(real_focus)
plt.fill_between(
    range(len(real_focus)),
    real_focus - 22,
    real_focus + 22,
    alpha=0.2,
    label="Axial resolution (±22 µm)"
)

plt.fill_between(
    range(len(real_focus)),
    real_focus - 5,
    real_focus + 5,
    alpha=0.3,
    label="Cellular resolution (±5 µm)"
)

plt.plot(
    dists,
    label="Inferred",
    color="blue",
    linewidth=2
)


plt.axhline(
    y=0,
    color="black",
    linestyle="--"
)


plt.xlabel(
    "Image Index",
    fontsize=16
)

plt.ylabel(
    "Imaging to Focal Distance (microns)",
    fontsize=16
)
plt.ylim(-300, 300)     # change depending on stack focal range
plt.xlim(0, len(dists) - 1)
plt.legend(
    fontsize=14,
    loc="lower right"
)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

dists = np.array(dists)

# Calculate MAE
mae = np.mean(np.abs(real_focus - dists))

# Calculate classification accuracy
true_direction = np.where(real_focus >= 0, 1, 0)
pred_direction = np.where(dists >= 0, 1, 0)

classification_accuracy = np.mean(
    true_direction == pred_direction
) * 100

# Absolute focal-position error
errors = np.abs(dists - real_focus)

# Percentage of predictions within ±22 µm of ground truth
within_dof = errors <= 22

percentage_within_dof = np.mean(within_dof) * 100

within_res = errors <= 5

percentage_within_res = np.mean(within_res) * 100

print(
    f"Percentage within ±22 µm: "
    f"{percentage_within_dof:.2f}%"
)
# Add metrics to plot
plt.text(
    0.02, 0.95,
    f"MAE = {mae:.2f} µm\n"
    f"Classification accuracy = {classification_accuracy:.1f}%\n"
    f"Within ±5 µm = {percentage_within_res:.1f}%\n"
    f"Within ±22 µm = {percentage_within_dof:.1f}%",
    transform=plt.gca().transAxes,
    fontsize=13,
    verticalalignment="top",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.8
    }
)

plt.tight_layout()
#plt.savefig("ModelD_fish1.png")
#plt.savefig("evaluation_results_ModelA_s42/focus_6z2_s2_700epochs.png")
#plt.savefig("dleet.png")
print("Done: png saved")

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import tifffile as tiff
# include z dist between each frame if known

# -------------------------
# IMAGE PROCESSING
# -------------------------
def process_image(image, sat_prctile=99):
    sat = np.percentile(image, sat_prctile)
    image = sat * np.tanh(image / sat)
    image = image / np.sqrt(np.sum(image**2) + 1e-8)
    return image


# -------------------------
# PATCH SELECTION
# -------------------------
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

            patch = image[x:x+patch_size, y:y+patch_size]

            if np.percentile(patch, 99) > threshold:
                patches.append((x, y))
                break

    return patches


# -------------------------
# LOAD TIFF STACK
# -------------------------
tiff_path = "session2/stack/uclaminiscopev4-stack_1_40fps.tif"   

images = tiff.imread(tiff_path)

print("Loaded stack shape:", images.shape)
#  (201, 608, 608)


# -------------------------
# MODEL SETUP
# -------------------------
model = MobileNetV2()

model.features[0][0] = Conv2dNormActivation(
    1, 32, kernel_size=1, norm_layer=None, activation_layer=nn.ReLU
)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 3),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model.to(device)

checkpoint_path = "/session2/stack/model_checkpoint_epoch_40.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()


# -------------------------
# VIDEO SETUP
# -------------------------
writer = FFMpegWriter(fps=15)

fig = plt.figure(facecolor='0.9', figsize=(14, 14), dpi=100)
gs = fig.add_gridspec(nrows=9, ncols=9, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.1)

ax0 = fig.add_subplot(gs[0:8, 0:7])
ax1 = fig.add_subplot(gs[0:4, 7:])


# -------------------------
# PARAMETERS
# -------------------------
offset = 0
real_focus = []
dists = []

img_range = range(5, 55)

patch_size = 250


# -------------------------
# MAIN LOOP
# -------------------------
with writer.saving(fig, "focus_video.mp4", 100):

    for img in img_range:
        print("Frame:", img)

        this_image = images[img]

        # display image
        ax0.clear()
        ax0.imshow(this_image, interpolation='nearest',
                   vmin=0, vmax=50, cmap='gray')
        ax0.set_xticks([])
        ax0.set_yticks([])

        # REAL FOCUS (frame-relative, NOT microns)
        real_focus.append(img - images.shape[0] // 2)

        # select patches
        selected_patches = select_patches(
            this_image, patch_size, 128, 25
        )

        all_patches = []

        for (x, y) in selected_patches:
            patch = this_image[x:x+patch_size, y:y+patch_size].astype(np.float32)

            # 2x2 binning
            patch = patch.reshape(
                patch.shape[0] // 2, 2,
                patch.shape[1] // 2, 2
            ).sum(axis=(1, 3))

            patch = process_image(patch, sat_prctile=95)

            all_patches.append(patch)

        all_patches = np.array(all_patches)

        all_patches = transforms.ToImage()(all_patches)
        all_patches = all_patches.permute(1, 2, 0).unsqueeze(1)
        all_patches = all_patches.to(device)

        # -------------------------
        # MODEL INFERENCE
        # -------------------------
        with torch.no_grad():
            pred = model(all_patches).cpu().numpy()

        mean_distance = np.mean(pred[:, 0] * 40)

        mean_class = np.mean(np.argmax(pred[:, 1:], axis=1))

        if mean_class < 0.5:
            mean_distance *= -1

        dists.append(mean_distance)

        # -------------------------
        # PLOT RESULTS
        # -------------------------
        ax1.clear()
        ax1.plot(real_focus, label='Real (frame index)', color='k', linewidth=3)
        ax1.plot(dists, label='Predicted', color='b', linewidth=2)

        ax1.axhline(y=real_focus[-1], color='r', linestyle='--')
        ax1.axhline(y=0, color='k', linestyle='--')

        ax1.set_xlim(0, len(list(img_range)))
        ax1.set_ylim(-15, 15)

        ax1.set_xlabel('Frame Index', fontsize=16)
        ax1.set_ylabel('Focus (relative)', fontsize=16)
        ax1.legend(fontsize=14, loc='lower right')

        ax1.tick_params(axis='both', which='major', labelsize=14)

        writer.grab_frame()

writer.finish()

print("Done: focus_video.mp4 saved")
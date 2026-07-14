import numpy as np
import os
from PIL import Image
from scipy.signal import medfilt2d
import tifffile as tiff
import matplotlib.pyplot as plt

# OUTPUT FOLDER

subdir = "session2/stack"
os.makedirs(subdir, exist_ok=True)

np.random.seed(0)


# IMAGE PROCESSING

def process_image(image, sat_prctile=99):
    sat = np.percentile(image, sat_prctile)
    image = sat * np.tanh(image / (sat + 1e-8))
    image = image / np.sqrt(np.sum(image ** 2) + 1e-8)
    return image



# PATCH SELECTION

def select_patches(image, patch_size, num_patches, threshold):

    patches = []

    x_min = image.shape[0] // 4
    x_max = image.shape[0] * 3 // 4

    y_min = image.shape[1] // 4
    y_max = image.shape[1] * 3 // 4

    while len(patches) < num_patches:

        x = np.random.randint(x_min, x_max - patch_size)
        y = np.random.randint(y_min, y_max - patch_size)

        patch = image[x:x + patch_size, y:y + patch_size]

        if np.percentile(patch, 99) > threshold:
            patches.append((x, y))

    return patches



# FOCUS ESTIMATION

def find_focus_dists(image_stack, distance_between_images=1.0, debug=False):

    uc = np.zeros(image_stack.shape[0])

    for i, image in enumerate(image_stack):

        image = process_image(image.astype(np.float32), sat_prctile=99)

        # Faster than full FFT because signal is real-valued
        flt = np.mean(image, axis=0)
        spectrum = np.abs(np.fft.rfft(flt)) ** 2

        uc[i] = -np.std(spectrum)

    best_focus = np.argmax(uc)

    if debug:


        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        axs[0].plot(uc)
        axs[0].set_title("Focus score")

        axs[1].imshow(image_stack[best_focus], cmap="gray")
        axs[1].set_title("Best focus")

        plt.show()

    focus_dists = (
        np.arange(image_stack.shape[0]) - best_focus
    ) * distance_between_images

    return best_focus, focus_dists



# LOAD TIFF STACK

def load_tiff_stack(path):

    stack = tiff.imread(path)

    print("Loaded TIFF stack:", stack.shape)

    return stack



# MAIN EXTRACTION PIPELINE

def extract_patches_from_tiff(tiff_path, patch_size, num_patches, threshold):

    stack = load_tiff_stack(tiff_path)

    file_names = []
    focal_distances = []

    file_id = os.path.splitext(os.path.basename(tiff_path))[0]

    middle_image = stack[stack.shape[0] // 2]

    print("Selecting patches...")

    # Only done ONCE instead of 101 times
    selected_patches = select_patches(
        middle_image,
        patch_size,
        num_patches,
        threshold
    )

    print(f"{len(selected_patches)} patches selected.")

    plt.figure(figsize=(8,8))
    plt.imshow(middle_image,cmap='gray')

    for x,y in selected_patches:

        plt.gca().add_patch(
            plt.Rectangle(
            (y,x),
            70, # originally 96 x 96
            70,
            edgecolor='red',
            fill=False,
            linewidth=0.5
            )
        )

    plt.savefig("selected_patches.png")

    for patch_idx, (x, y) in enumerate(selected_patches):

        print(f"Processing patch {patch_idx + 1}/{len(selected_patches)}")

        patch_stack = stack[:, x:x + patch_size, y:y + patch_size]

        # Compute focus ONCE for this patch
        best_focus, focus_dists = find_focus_dists(
            patch_stack,
            distance_between_images=1.0
        )

        # Skip patches whose focus is too close to stack boundaries
        if best_focus < 30 or best_focus > stack.shape[0] - 30: # maybe change this logic
            print("Skipping patch (focus near edge).")
            continue

        # Generate every distance from the same focus calculation
        for dist in range(-100, 101): #change to -100 to 100 originally -13 to 13

            frame_idx = np.argmin(np.abs(focus_dists - dist))

            image = patch_stack[frame_idx].astype(np.float32)

            # 2x2 binning
            image = image.reshape(
                image.shape[0] // 2,
                2,
                image.shape[1] // 2,
                2
            ).sum(axis=(1, 3))

            if np.isnan(image).any() or np.isinf(image).any():
                continue

            image = process_image(image, sat_prctile=95)

            out_name = f"{file_id}_patch{patch_idx}_d{dist}.tif"
            out_path = os.path.join(subdir, out_name)

            Image.fromarray(image).save(out_path)

            file_names.append(out_name)
            focal_distances.append(dist)

    return file_names, np.array(focal_distances)



# RUN SCRIPT

if __name__ == "__main__":

    tiff_path = "session2/stack/uclaminiscopev4-stack_1_40fps.tif"

    patch_size = 70 # originally 250
    threshold = 25
    num_patches = 128 # originally 20 but too small

    file_names, distances = extract_patches_from_tiff(
        tiff_path,
        patch_size,
        num_patches,
        threshold
    )

    csv_path = os.path.join(subdir, "file_names_and_distances.csv")

    with open(csv_path, "w") as f:
        f.write("file_name,distance\n")
        for name, dist in zip(file_names, distances):
            f.write(f"{name},{dist}\n")

    print(f"\nFinished! Generated {len(file_names)} images.")
    print("CSV saved to:", csv_path)

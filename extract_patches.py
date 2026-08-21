import time
import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import os


# -------------------------
# SEED
# -------------------------
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


# -------------------------
# DATASET (FIXED FOR TIFF PATCHES)
# -------------------------
class FocusDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform

        self.sign = lambda x: int(x >= 0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = Image.open(img_path).convert("F")  # IMPORTANT: grayscale float

        label_val = float(self.img_labels.iloc[idx, 1])

        # regression target (scaled)
        dist = abs(label_val / 40.0)

        # classification: 0 = negative, 1 = positive
        sign = self.sign(label_val)

        if self.transform:
            image = self.transform(image)

        # return tensors
        label = (
            torch.tensor(dist, dtype=torch.float32),
            torch.tensor(sign, dtype=torch.long)
        )

        return image, label


# -------------------------
# TRANSFORMS
# -------------------------
tr = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomAffine(
    degrees=(-30, 30),  # Rotation range
    translate=(0.1, 0.1),  # Translation range (as a fraction of total size)
    scale=(0.8, 1.2),  # Scale range
    shear=(-10, 10)  # Shear range
    )
    #transforms.ToDtype(torch.float32, scale=True)
    #transforms.Resize((224, 224)),
    # transforms.RandomRotation(90)
])
# Need to add translations, maybe even brightness?


# -------------------------
# LOAD DATA
# -------------------------
all_data = FocusDataset(
    #'output/file_names_and_distances_combined.csv',    
    #    'session2/stack_224_128/file_names_and_distances.csv', #if using just one tiff file
    #  'session2/stack_224_128',
    #'output',
    # 'session3/zStack_224_128/file_names_and_distances.csv',
    # 'session3/zStack_224_128',
    #'patches_40/file_names_and_distances_combined.csv',
    #'patches_40',
    'five_fish_patches/file_names_and_distances_five_fish.csv',
    'five_fish_patches',
    transform=tr
)

print("Dataset size:", len(all_data))


# -------------------------
# VISUAL CHECK
# -------------------------
#indices = np.random.choice(len(all_data), 25, replace=False)

#fig, axs = plt.subplots(5, 5)

#for i, idx in enumerate(indices):
 #   img = all_data[idx][0][0].numpy()
#    axs[i//5, i%5].imshow(img, cmap='gray')

#plt.show()
#plt.savefig("dataset_visualization_combined.png", dpi=300)

# -------------------------
# MODEL
# -------------------------
model = MobileNetV2()

model.features[0][0] = Conv2dNormActivation(
    1, 32,
    kernel_size=1,
    norm_layer=None,
    activation_layer=nn.ReLU
)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1280, 3),
)


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
# indices_test = [
#     i for i, s in enumerate(all_data.img_labels.iloc[:, 0])
#     if 'f3_plane2' in s
# ]
# -------------------------
# SAFE TRAIN/TEST SPLIT
# -------------------------
num_samples = len(all_data)
#num_samples = 20

indices = np.arange(num_samples)
np.random.shuffle(indices)

split = int(0.8 * num_samples)

indices_train = indices[:split]
indices_test = indices[split:]

train_data = torch.utils.data.Subset(all_data, indices_train)
test_data = torch.utils.data.Subset(all_data, indices_test)

print("Train size:", len(train_data))
print("Test size:", len(test_data))

test_data = torch.utils.data.Subset(all_data, indices_test)
train_data = torch.utils.data.Subset(
    all_data,
    [i for i in range(len(all_data)) if i not in indices_test]
)


train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True) # initial batch size 16


# -------------------------
# LOSS / OPTIMIZER
# -------------------------
loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)# add weight decay = 0
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  
scheduler = MultiStepLR(optimizer,milestones=[10, 100, 150], gamma=0.5) # add milestones


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train(dataloader, model):
    model.train()

    reg_loss_total = 0
    class_loss_total = 0

    for X, y in dataloader:
        print("Training")
        X = X.to(device)

        y_reg = y[0].to(device)
        y_class = y[1].to(device)

        pred = model(X)

        loss_reg = loss_fn_reg(pred[:, 0], y_reg)
        loss_class = loss_fn_class(pred[:, 1:], y_class)

        loss = loss_reg + loss_class

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reg_loss_total += loss_reg.item()
        class_loss_total += loss_class.item()

    return reg_loss_total / len(dataloader), class_loss_total / len(dataloader)


# -------------------------
# TEST FUNCTION
# -------------------------
def test(dataloader, model):
    model.eval()
    print("Testing")
    reg_loss_total = 0
    class_loss_total = 0

    all_dist_labels = []
    all_dist_preds = []
    all_sign_labels = []
    all_sign_preds = []
    all_images = []
    all_filenames = []

    all_true_distances = []
    all_pred_distances = []

    all_true_signs = []
    all_pred_signs = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_reg = y[0].to(device)
            y_class = y[1].to(device)

            pred = model(X)

            reg_loss_total += loss_fn_reg(pred[:, 0], y_reg).item()
            class_loss_total += loss_fn_class(pred[:, 1:], y_class).item()

            # Saving predictions
            pred_abs_distance = pred[:, 0] * 40
            # Scaled by 1/40 so descaling...
            true_abs_distance = y_reg * 40

            # Direction
            pred_sign = torch.argmax(
                pred[:, 1:],
                dim=1
            )
            true_sign = y_class

            # COMBINE MAGNITUDE + SIGN

            pred_signed_distance = torch.where(
                pred_sign == 0,
                -pred_abs_distance,
                pred_abs_distance
            )

            true_signed_distance = torch.where(
                true_sign == 0,
                -true_abs_distance,
                true_abs_distance
            )


            all_dist_labels.append(y_reg.cpu().numpy())
            all_dist_preds.append(pred[:, 0].cpu().numpy())

            all_sign_labels.append(y_class.cpu().numpy())
            all_sign_preds.append(torch.argmax(pred[:, 1:], dim=1).cpu().numpy())



            all_true_distances.extend(
                true_signed_distance.cpu().numpy()
            )

            all_pred_distances.extend(
                pred_signed_distance.cpu().numpy()
            )

            all_true_signs.extend(
                true_sign.cpu().numpy()
            )

            all_pred_signs.extend(
                pred_sign.cpu().numpy()
            )

            # Save images for visualisation
            all_images.extend(
                X.cpu().numpy()
            )

    all_true_distances = np.array(
        all_true_distances
    )

    all_pred_distances = np.array(
        all_pred_distances
    )

    all_true_signs = np.array(
        all_true_signs
    )

    all_pred_signs = np.array(
        all_pred_signs
    )

    mae = mean_absolute_error(
        all_true_distances,
        all_pred_distances
    )

    rmse = np.sqrt(
        mean_squared_error(
            all_true_distances,
            all_pred_distances
        )
    )

    r2 = r2_score(
        all_true_distances,
        all_pred_distances
    )


    # =====================================================
    # CALCULATE CLASSIFICATION ACCURACY
    # =====================================================

    accuracy = accuracy_score(
        all_true_signs,
        all_pred_signs
    )



# missing batch code if batch == 0; else
    return (
        reg_loss_total / len(dataloader),
        class_loss_total / len(dataloader),
        mae,
        rmse,
        r2,
    accuracy
   # all_true_distances,
   # all_pred_distances,
   # all_true_signs,
   # all_pred_signs
    )


# -------------------------
# TRAIN LOOP
# -------------------------
epochs = 1020 # 100 produces smooth curve, stabilises after 10 epochs but keep it at 200 to be safe
train_reg_losses = []
train_class_losses = []
test_reg_losses = []
test_class_losses = []
test_mae = []
test_rmse = []
test_r2 = []
test_accuracy = []

# # load checkpoint if it exists:
# # find potential checkpoint files:
# checkpoint_files = [f for f in os.listdir('.') if f.startswith('model_checkpoint_epoch_') and f.endswith('.pth')]
# last_epoch = 0
# if checkpoint_files:
#     # sort by epoch number and take the last one:
#     checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
#     checkpoint_path = checkpoint_files[-1]
#     print(f'Loading checkpoint from {checkpoint_path}')
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     # also update the scheduler to start from the last epoch:
#     last_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
#     scheduler.last_epoch = last_epoch

start_time = time.time()
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    train_reg, train_class = train(train_dataloader, model)
    #test_reg, test_class = test(test_dataloader, model)

    (
        test_reg,
        test_class,
        mae,
        rmse,
        r2,
        accuracy
       # true_distances,
       # pred_distances,
       # true_signs,
       # pred_signs
    ) = test(
        test_dataloader,
        model
    )

    train_reg_losses.append(train_reg)
    train_class_losses.append(train_class)
    test_reg_losses.append(test_reg)
    test_class_losses.append(test_class)
    test_mae.append(mae)
    test_rmse.append(rmse)
    test_r2.append(r2)
    test_accuracy.append(accuracy)
    scheduler.step()

    print(f"Train reg: {train_reg:.4f}, class: {train_class:.4f}")
    print(f"Test reg: {test_reg:.4f}, class: {test_class:.4f}")


    if epoch % 100 == 0: #change logic to epoch+1
        torch.save(model.state_dict(), f"model_combined_7datasets_patches_checkpoint_1020epochs_{epoch}.pth")

end_time = time.time()
training_time = end_time - start_time

plt.figure(figsize=(10, 6))

plt.plot(test_mae, label="MAE")
plt.plot(test_rmse, label="RMSE")

plt.xlabel("Epoch")
plt.ylabel("Focal distance error")
plt.title("Regression performance")
plt.legend()
plt.grid(True)

plt.savefig("regression_performance_1000epochs.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))

plt.plot(test_accuracy, label="Classification accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classification performance")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.savefig("classification_accuracy_1000epochs.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))

plt.plot(test_r2)

plt.xlabel("Epoch")
plt.ylabel("$R^2$")
plt.title("Regression $R^2$ over training")
plt.grid(True)

plt.savefig("r2_over_epochs_1000epochs.png", dpi=300)
#plt.show()

csv_file = "five_fish_patches/file_names_and_distances_five_fish.csv"
image_folder = "five_fish_patches"

df = pd.read_csv(csv_file)

df["patch_id"] = (
    df["file_name"]
    .str.extract(r"patch(\d+)")[0]
    .astype(int)
)


# -----------------------------------------------------
# SELECT THREE RANDOM PATCHES
# -----------------------------------------------------

reference_z = 0

reference_candidates = df[
    df["distance"] == reference_z
]

selected_patch_ids = (
    reference_candidates[
        "patch_id"
    ]
    .sample(
        n=3,
        random_state=42
    )
    .tolist()
)

print(
    "Selected patches:",
    selected_patch_ids
)

df["patch_id"] = (
    df["file_name"]
    .str.extract(r"patch(\d+)")[0]
    .astype(int)
)



# SELECT THREE RANDOM PATCHES


reference_z = 0

reference_candidates = df[
    df["distance"] == reference_z
]

selected_patch_ids = (
    reference_candidates[
        "patch_id"
    ]
    .sample(
        n=3,
        random_state=42
    )
    .tolist()
)

print(
    "Selected patches:",
    selected_patch_ids
)
distances = list(
    range(10, -10, -5) # change back to maybe 20 for 11 images originally 100, -101, -20
)#100,-101,-20


# =====================================================
# MODEL PREDICTION FUNCTION
# =====================================================

def predict_distance(image):

    # Convert image to tensor
    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    # Add channel dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to GPU/CPU
    image_tensor = image_tensor.to(device)

    # Model prediction
    with torch.no_grad():

        pred = model(
            image_tensor
        )

    # Regression magnitude
    predicted_magnitude = (
        pred[0, 0].item() * 40
    )

    # Classification sign
    predicted_class = torch.argmax(
        pred[:, 1:],
        dim=1
    ).item()

    # Combine magnitude and sign
    if predicted_class == 0:

        predicted_distance = (
            -predicted_magnitude
        )

    else:

        predicted_distance = (
            predicted_magnitude
        )

    return predicted_distance


# =====================================================
# DISPLAY IMAGE FUNCTION
# =====================================================

def display_image(
    img,
    percentile=99
):

    img = img.astype(
        np.float32
    )

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


# =====================================================
# PLOT
# =====================================================

n_rows = len(distances)
n_cols = 3

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        12,
        n_rows * 2.5
    )
)

axes = np.atleast_2d(
    axes
)


for i, d in enumerate(
    distances
):

    for j, patch_id in enumerate(
        selected_patch_ids
    ):

        # Find image in CSV
        match = df[
            (df["distance"] == d) &
            (df["patch_id"] == patch_id)
        ]

        if len(match) == 0:

            axes[i, j].axis(
                "off"
            )

            continue


        # Get row
        row = match.iloc[0]


        # Get filename
        filename = row[
            "file_name"
        ]


        # Get frame number


        # Load image
        path = os.path.join(
            image_folder,
            filename
        )

        img = np.array(
            Image.open(path)
        )


        # Display image
        img_display, vmin, vmax = (
            display_image(img)
        )


        axes[i, j].imshow(
            img_display,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )


        # =================================================
        # PREDICT FOCAL DISTANCE
        # =================================================

        predicted_distance = (
            predict_distance(img)
        )


        # True focal distance
        true_distance = row[
            "distance"
        ]


        # =================================================
        # TITLE
        # =================================================

        axes[i, j].set_title(
            f"True: {true_distance:+.0f} µm\n"
            f"Predicted: {predicted_distance:+.1f} µm\n",
            fontsize=9
        )


        axes[i, j].axis(
            "off"
        )


    # =====================================================
    # Z LABEL
    # =====================================================

    axes[i, 0].set_ylabel(
        f"z = {d:+.0f} µm",
        fontsize=11,
        rotation=0,
        labelpad=45,
        va="center"
    )


# =====================================================
# COLUMN LABELS
# =====================================================

for j, patch_id in enumerate(
    selected_patch_ids
):

    axes[0, j].set_title(
        f"Patch {patch_id}",
        fontsize=12
    )


# =====================================================
# LAYOUT
# =====================================================

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.95,
    bottom=0.03,
    hspace=0.4,
    wspace=0.05
)


# SAVE


plt.savefig(
    "three_patch_predictions_7datasets_1000epochs.png",
    dpi=300,
    bbox_inches="tight"
)

h = training_time / 3600
m = training_time / 60
s = m % 60
print(f"Training time: {h:.0f} hours, {m:.0f} minutes, {s:.0f} seconds or {training_time:.2f} seconds")
plt.figure(figsize=(10, 6))

plt.plot(train_reg_losses, label='Train Regression')
plt.plot(test_reg_losses, label='Test Regression')
plt.plot(train_class_losses, label='Train Classification')
plt.plot(test_class_losses, label='Test Classification')

plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training and Test Loss")
plt.yscale("log")
plt.legend()
plt.grid(True, which = "both", linestyle = "--", alpha = 0.5)
plt.savefig("loss_plot_7datasets_1000epochs.png", dpi = 300, bbox_inches="tight") 

#plt.show()
print("DONE")

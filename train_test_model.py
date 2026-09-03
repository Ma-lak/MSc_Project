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



# SEED

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)



# DATASET 
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

        image = Image.open(img_path).convert("F")  
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



# TRANSFORMS
tr = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomAffine(
    degrees=(-30, 30),  # Rotation range
    translate=(0.1, 0.1),  # Translation range 
    scale=(0.8, 1.2),  # Scale range
    shear=(-10, 10)  # Shear range
    )
])

tr_test = transforms.Compose([
    transforms.ToImage()
])




# LOAD DATA

all_data = FocusDataset(
    'modelA_patches/file_names_and_distances_modelA.csv',
    'modelA_patches',
    transform=tr
)

test_fish = FocusDataset(
    "test_patches/file_names_and_distances_modelA.csv",
    "test_patches",
    transform=tr_test
)


print("Dataset size:", len(all_data))


# MODEL
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


# TRAIN / TEST SPLIT


train_data = all_data
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(
    test_fish,
    batch_size=128,
    shuffle=False
    )


#train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True) 



# LOSS / OPTIMISER
loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.CrossEntropyLoss()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
scheduler = MultiStepLR(optimizer,milestones=[10, 100, 150], gamma=0.5) 


# TRAIN FUNCTION
def train(dataloader, model):
    model.train()

    reg_loss_total = 0
    class_loss_total = 0

    for X, y in dataloader:
        #print("Training")
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


# TEST FUNCTION
def test(dataloader, model):
    model.eval()
    print("Testing")
    reg_loss_total = 0
    class_loss_total = 0

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


    # CALCULATE CLASSIFICATION ACCURACY

    accuracy = accuracy_score(
        all_true_signs,
        all_pred_signs
    )



    return (
        reg_loss_total / len(dataloader),
        class_loss_total / len(dataloader),
        mae,
        rmse,
        r2,
    accuracy
    )



# TRAIN LOOP
epochs = 810 
train_reg_losses = []
train_class_losses = []
test_reg_losses = []
test_class_losses = []
test_mae = []
test_rmse = []
test_r2 = []
test_accuracy = []


start_time = time.time()
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    start_time1 = time.time()

    train_reg, train_class = train(train_dataloader, model)
    train_reg_losses.append(train_reg)
    train_class_losses.append(train_class)
    test_reg, test_class = test(test_dataloader, model)
    # OPTIONAL
##    if (epoch + 1) % 100 == 0:
##        test_reg, test_class, mae, rmse, r2, accuracy = test(
##            test_dataloader,
##            model
##        )
##        test_reg_losses.append(test_reg)
##        test_class_losses.append(test_class)
##
##
##        test_epochs.append(epoch + 1)
##
##        test_mae.append(mae)
##        test_rmse.append(rmse)
##        test_r2.append(r2)
##        test_accuracy.append(accuracy)
##        print(f"Test reg: {test_reg:.4f}, class: {test_class:.4f}")
    scheduler.step()
    end_time1 = time.time()
    print(f"Train reg: {train_reg:.4f}, class: {train_class:.4f}")
    print("Training time in seconds: ", end_time1 - start_time1)



    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"modelA_s42_bs128_checkpoint_{epoch}.pth")

end_time = time.time()
training_time = end_time - start_time

plt.figure(figsize=(10, 6))

plt.plot(test_epochs, test_mae, label="MAE")
plt.plot(test_epochs, test_rmse, label="RMSE")

plt.xlabel("Epoch")
plt.ylabel("Focal distance error (µm)")
plt.legend()
plt.grid(True)

plt.savefig("regression_performance_1000epochs_modelA_s42_bs128.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))

plt.plot(test_epochs, test_accuracy)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Classification performance")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.savefig("classification_accuracy_1000epochs_modelA_s42_bs128.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))

plt.plot(test_epochs, test_r2)

plt.xlabel("Epoch")
plt.ylabel("$R^2$")
plt.title("Regression $R^2$ over training")
plt.grid(True)

plt.savefig("r2_over_epochs_1000epochs_modelA_s42_bs128.png", dpi=300)
#plt.show()

plt.figure(figsize=(10, 6))

plt.plot(range(1, epochs + 1), train_reg_losses, label="Train Regression")
plt.plot(test_epochs, test_reg_losses, label="Test Regression")

plt.plot(range(1, epochs + 1), train_class_losses, label="Train Classification")
plt.plot(test_epochs, test_class_losses, label="Test Classification")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
#plt.yscale("log")
plt.legend()
plt.grid(True, which = "both", linestyle = "--", alpha = 0.5)
plt.savefig("loss_plot_modelA_1000epochs_s42_bs128.png", dpi = 300, bbox_inches="tight")


print(f"Training time: {training_time:.2f} seconds")




csv_file = "modelA_patches/file_names_and_distances_modelA.csv"
image_folder = "modelA_patches"

df = pd.read_csv(csv_file)




## SELECT THREE RANDOM PATCHES


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



### SELECT THREE RANDOM PATCHES


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
##
##
### 
### MODEL PREDICTION FUNCTION
###
##
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


##        # =================================================
##        # PREDICT FOCAL DISTANCE
##        # =================================================

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
    "three_patch_predictions_modelA.png",
    dpi=300,
    bbox_inches="tight"
)



#plt.show()
print("DONE")


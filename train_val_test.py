import time
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)


from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR



# PARAMETERS
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
BATCH_SIZE = 128
EPOCHS = 270



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
    translate=(0.1, 0.1),  # Translation range (as a fraction of total size)
    scale=(0.8, 1.2),  # Scale range
    shear=(-10, 10)  # Shear range
    )

])


tr_test = transforms.Compose([
    transforms.ToImage()
])


all_data = FocusDataset(
    'modelA_patches/file_names_and_distances_train.csv',
    'modelA_patches',
    transform=tr
)

train_data = all_data

##test_dataloader = DataLoader(
##    test_fish,
##    batch_size=128,
##    shuffle=False)


# ALternative logic to split randomly (slice-level splitting)
##num_samples = len(all_data)
###num_samples = 20
##
##indices = np.arange(num_samples)
##np.random.shuffle(indices)
##
##train_end = int(0.70 * num_samples)
##val_end = int(0.85 * num_samples)
##
##indices_train = indices[:train_end]
##indices_val = indices[train_end:val_end]
##indices_test = indices[val_end:]

##train_data = torch.utils.data.Subset(all_data, indices_train)
##val_fish = torch.utils.data.Subset(all_data, indices_val)
##test_fish = torch.utils.data.Subset(all_data, indices_test)
##
##print("Train:", len(train_data))
##print("Validation:", len(val_fish))
##print("Test:", len(test_fish))

val_fish = FocusDataset(
    "modelA_patches/file_names_and_distances_val.csv",
    "modelA_patches",
    transform=tr_test
)

test_fish = FocusDataset(
    "test_patches/file_names_and_distances_modelA.csv",
    "test_patches",
    transform=tr_test
)


train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_fish,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_fish,
    batch_size=BATCH_SIZE,
    shuffle=False
)

loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.CrossEntropyLoss()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------

def create_model():

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

    return model.to(device)


model = create_model()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[10, 100, 150],
    gamma=0.5
)

def train_one_epoch():

    model.train()

    reg_loss_total = 0
    class_loss_total = 0

    true_distances = []
    pred_distances = []

    true_signs = []
    pred_signs = []

    for X, y in train_dataloader:

        X = X.to(device, non_blocking=True)
        y_reg = y[0].to(device, non_blocking=True)
        y_class = y[1].to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(X)

        loss_reg = loss_fn_reg(
            pred[:, 0],
            y_reg
        )

        loss_class = loss_fn_class(
            pred[:, 1:],
            y_class
        )

        loss = loss_reg + loss_class

        loss.backward()
        optimizer.step()

        reg_loss_total += loss_reg.item()
        class_loss_total += loss_class.item()

        # -------------------------
        # Calculate training MAE
        # -------------------------

        pred_abs = pred[:, 0] * 40
        true_abs = y_reg * 40

        pred_sign = torch.argmax(
            pred[:, 1:],
            dim=1
        )

        true_sign = y_class

        pred_signed = torch.where(
            pred_sign == 0,
            -pred_abs,
            pred_abs
        )

        true_signed = torch.where(
            true_sign == 0,
            -true_abs,
            true_abs
        )

        true_distances.extend(
            true_signed.detach().cpu().numpy()
        )

        pred_distances.extend(
            pred_signed.detach().cpu().numpy()
        )

        true_signs.extend(
            true_sign.detach().cpu().numpy()
        )

        pred_signs.extend(
            pred_sign.detach().cpu().numpy()
        )

    # -------------------------
    # Training metrics
    # -------------------------

    train_mae = mean_absolute_error(
        true_distances,
        pred_distances
    )

    train_accuracy = accuracy_score(
        true_signs,
        pred_signs
    )

    return (
        reg_loss_total / len(train_dataloader),
        class_loss_total / len(train_dataloader),
        train_mae,
        train_accuracy
    )



# VALIDATION

def evaluate(loader):

    model.eval()

    reg_loss_total = 0
    class_loss_total = 0

    true_distances = []
    pred_distances = []

    true_signs = []
    pred_signs = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y_reg = y[0].to(device)
            y_class = y[1].to(device)

            pred = model(X)

            loss_reg = loss_fn_reg(
                pred[:, 0],
                y_reg
            )

            loss_class = loss_fn_class(
                pred[:, 1:],
                y_class
            )

            reg_loss_total += loss_reg.item()
            class_loss_total += loss_class.item()

            # Magnitude
            pred_abs = pred[:, 0] * 40
            true_abs = y_reg * 40

            # Direction
            pred_sign = torch.argmax(
                pred[:, 1:],
                dim=1
            )

            true_sign = y_class

            # Signed distance
            pred_signed = torch.where(
                pred_sign == 0,
                -pred_abs,
                pred_abs
            )

            true_signed = torch.where(
                true_sign == 0,
                -true_abs,
                true_abs
            )

            true_distances.extend(
                true_signed.cpu().numpy()
            )

            pred_distances.extend(
                pred_signed.cpu().numpy()
            )

            true_signs.extend(
                true_sign.cpu().numpy()
            )

            pred_signs.extend(
                pred_sign.cpu().numpy()
            )

    true_distances = np.array(
        true_distances
    )

    pred_distances = np.array(
        pred_distances
    )

    true_signs = np.array(
        true_signs
    )

    pred_signs = np.array(
        pred_signs
    )

    mae = mean_absolute_error(
        true_distances,
        pred_distances
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_distances,
            pred_distances
        )
    )

    r2 = r2_score(
        true_distances,
        pred_distances
    )

    accuracy = accuracy_score(
        true_signs,
        pred_signs
    )

    return (
        reg_loss_total / len(loader),
        class_loss_total / len(loader),
        mae,
        rmse,
        r2,
        accuracy
    )



train_reg_losses = []
train_class_losses = []

val_reg_losses = []
val_class_losses = []

train_mae = []
train_accuracy = []

val_mae = []
val_rmse = []
val_r2 = []
val_accuracy = []





# TRAINING LOOP

start_time = time.time()

for epoch in range(EPOCHS):
    print("Epoch ", epoch + 1)

    (
    train_reg,
    train_class,
    train_mae_value,
    train_accuracy_value
    ) = train_one_epoch() 

    # Validation every epoch
    (
        val_reg,
        val_class,
        mae,
        rmse,
        r2,
        accuracy
    ) = evaluate(val_loader)

    train_reg_losses.append(
        train_reg
    )

    train_class_losses.append(
        train_class
    )

    train_mae.append(
    train_mae_value
    )

    train_accuracy.append(
    train_accuracy_value
    )

    val_reg_losses.append(
        val_reg
    )

    val_class_losses.append(
        val_class
    )

    val_mae.append(
        mae
    )

    val_rmse.append(
        rmse
    )

    val_r2.append(
        r2
    )

    val_accuracy.append(
        accuracy
    )

    scheduler.step()

    if epoch % 10 == 0:
        print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train MAE: {train_mae_value:.2f} µm | "
        f"Val MAE: {mae:.2f} µm | "
        f"Train Acc: {train_accuracy_value*100:.2f}% | "
        f"Val Acc: {accuracy*100:.2f}%"
        )

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"model_val_checkpoint_epoch_{epoch+1}.pth")



# FINAL TEST — ONLY ONCE - could change to specific epoch


print("\nFinal evaluation on unseen test fish...")

(
    test_reg,
    test_class,
    test_mae,
    test_rmse,
    test_r2,
    test_accuracy
) = evaluate(test_loader)


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    f"Regression loss: {test_reg:.4f}"
)

print(
    f"Classification loss: {test_class:.4f}"
)

print(
    f"MAE: {test_mae:.2f} µm"
)

print(
    f"RMSE: {test_rmse:.2f} µm"
)

print(
    f"R²: {test_r2:.4f}"
)

print(
    f"Classification accuracy: "
    f"{test_accuracy*100:.2f}%"
)


# PLOT 1 — TRAIN VS VALIDATION LOSS

plt.figure(figsize=(10, 6))

plt.plot(
    train_reg_losses,
    label="Train regression"
)

plt.plot(
    val_reg_losses,
    label="Validation regression"
)

plt.plot(
    train_class_losses,
    label="Train classification"
)

plt.plot(
    val_class_losses,
    label="Validation classification"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "train_validation_losslr5.png",
    dpi=300
)

plt.close()


# PLOT 2 — MAE

plt.figure(figsize=(10, 6))

plt.plot(
    train_mae,
    label="Training MAE"
)

plt.plot(
    val_mae,
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE (µm)")
plt.title("Training vs Validation MAE")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "train_validation_maelr5.png",
    dpi=300
)

plt.close()


# PLOT 3 — CLASSIFICATION ACCURACY

plt.figure(figsize=(10, 6))

plt.plot(
    np.array(train_accuracy) * 100,
    label="Training accuracy"
)

plt.plot(
    np.array(val_accuracy) * 100,
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Classification accuracy (%)")
plt.title("Training vs Validation Classification Accuracy")

plt.ylim(0, 100)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "train_validation_accuracy.png",
    dpi=300
)

plt.close()



# TRAINING TIME


training_time = time.time() - start_time

print(
    f"\nTraining time: "
    f"{training_time / 3600:.2f} hours"
)

print("DONE")






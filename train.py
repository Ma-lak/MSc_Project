import time
import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms

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
    transforms.ToDtype(torch.float32, scale=True)
    #transforms.Resize((1000, 1000)),
    # transforms.RandomRotation(90)
])
# Need to add resize and rotations



loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.CrossEntropyLoss()


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train(dataloader, model):
    global optimizer
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

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_reg = y[0].to(device)
            y_class = y[1].to(device)

            pred = model(X)

            reg_loss_total += loss_fn_reg(pred[:, 0], y_reg).item()
            class_loss_total += loss_fn_class(pred[:, 1:], y_class).item()

            all_dist_labels.append(y_reg.cpu().numpy())
            all_dist_preds.append(pred[:, 0].cpu().numpy())

            all_sign_labels.append(y_class.cpu().numpy())
            all_sign_preds.append(torch.argmax(pred[:, 1:], dim=1).cpu().numpy())
# missing batch code if batch == 0; else
    return (
        reg_loss_total / len(dataloader),
        class_loss_total / len(dataloader)
    )

# -------------------------
# LOAD DATA
# -------------------------
all_data = FocusDataset(
    'combined_data/file_names_and_distances.csv',    
    #  'session2/stack/file_names_and_distances.csv', if using just one tiff file
    # 'session2/stack',
    'combined_data',
    transform=tr
)

print("Dataset size:", len(all_data))


# -------------------------
# VISUAL CHECK
# -------------------------
indices = np.random.choice(len(all_data), 25, replace=False)

fig, axs = plt.subplots(5, 5)

for i, idx in enumerate(indices):
    img = all_data[idx][0][0].numpy()
    axs[i//5, i%5].imshow(img, cmap='gray')

plt.show()



# -------------------------
# 100 RANDOM TRAIN/TEST SPLITS
# -------------------------
num_splits = 2
epochs = 20

all_train_reg = []
all_test_reg = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
start_time = time.time()

for split_num in range(num_splits):

    print(f"\n========== Split {split_num+1}/{num_splits} ==========")

    # -------------------------
    # Random train/test split
    # -------------------------
    indices = np.random.permutation(len(all_data))
    split = int(0.8 * len(all_data))

    train_idx = indices[:split]
    test_idx = indices[split:]

    train_data = torch.utils.data.Subset(all_data, train_idx)
    test_data = torch.utils.data.Subset(all_data, test_idx)

    train_dataloader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=False
    )

    # -------------------------
    # NEW MODEL
    # -------------------------
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

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5
    )

    # Make train() use the new optimizer
    globals()['optimizer'] = optimizer

    train_reg_losses = []
    test_reg_losses = []

    for epoch in range(epochs):

        train_reg, train_class = train(train_dataloader, model)
        test_reg, test_class = test(test_dataloader, model)

        train_reg_losses.append(train_reg)
        test_reg_losses.append(test_reg)

        scheduler.step()

    all_train_reg.append(train_reg_losses)
    all_test_reg.append(test_reg_losses)

# -------------------------
# TRAINING TIME
# -------------------------
end_time = time.time()

training_time = end_time - start_time

print(f"\nTotal training time = {training_time:.2f} seconds")

# -------------------------
# MEAN ± STD PLOT
# -------------------------
train_reg = np.array(all_train_reg)
test_reg = np.array(all_test_reg)

mean_train = train_reg.mean(axis=0)
std_train = train_reg.std(axis=0)

mean_test = test_reg.mean(axis=0)
std_test = test_reg.std(axis=0)

epochs_axis = np.arange(1, epochs + 1)

plt.figure(figsize=(12,8))

# Plot every run
for curve in train_reg:
    plt.plot(epochs_axis, curve,
             color='blue',
             alpha=0.05)

for curve in test_reg:
    plt.plot(epochs_axis, curve,
             color='red',
             alpha=0.05)

# Mean curves
plt.plot(
    epochs_axis,
    mean_train,
    color='blue',
    linewidth=3,
    label='Mean Train'
)

plt.plot(
    epochs_axis,
    mean_test,
    color='red',
    linewidth=3,
    label='Mean Test'
)

# Standard deviation bands
plt.fill_between(
    epochs_axis,
    mean_train - std_train,
    mean_train + std_train,
    color='blue',
    alpha=0.25
)

plt.fill_between(
    epochs_axis,
    mean_test - std_test,
    mean_test + std_test,
    color='red',
    alpha=0.25
)

plt.xlabel("Epoch")
plt.ylabel("Regression Loss")
plt.title("Regression Loss over 100 Random Train/Test Splits")
plt.grid(True)
plt.legend()

plt.savefig("regression_loss_100_splits_mean_std.png", dpi=300)
plt.show()

print("DONE")
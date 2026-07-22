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
    #transforms.ToDtype(torch.float32, scale=True)
    transforms.Resize((224, 224)),
    # transforms.RandomRotation(90)
])
# Need to add resize and rotations


# -------------------------
# LOAD DATA
# -------------------------
all_data = FocusDataset(
    #'output/file_names_and_distances_combined.csv',    
       'session2/stack_36_128/file_names_and_distances.csv', #if using just one tiff file
     'session2/stack_36_128',
    #'output',
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
# TRAIN LOOP
# -------------------------
epochs = 51 # 100 produces smooth curve, stabilises after 10 epochs but keep it at 20 to be safe
train_reg_losses = []
train_class_losses = []
test_reg_losses = []
test_class_losses = []

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
    test_reg, test_class = test(test_dataloader, model)

    train_reg_losses.append(train_reg)
    train_class_losses.append(train_class)
    test_reg_losses.append(test_reg)
    test_class_losses.append(test_class)
    scheduler.step()

    print(f"Train reg: {train_reg:.4f}, class: {train_class:.4f}")
    print(f"Test reg: {test_reg:.4f}, class: {test_class:.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"model_s2_size36__128_patches_50epochs_checkpoint_epoch_{epoch}.pth")


    # plt.draw()
    # plt.pause(0.001)
    # axs[1].cla()
    # axs[1].plot(train_reg_losses)
    # axs[1].plot(test_reg_losses)
    # plt.draw()
    # plt.pause(0.001)
    # axs[2].cla()
    # axs[2].plot(train_class_losses)
    # axs[2].plot(test_class_losses)
    # plt.draw()
    # plt.pause(0.001)
end_time = time.time()
training_time = end_time - start_time
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
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot__size36_128patches_batch40_50_epochs_s2.png") 

#plt.show()
print("DONE")

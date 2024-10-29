# %%
# import kagglehub
# # Download latest version
# path1 = kagglehub.dataset_download("nih-chest-xrays/data")
# # Download latest version
# path2 = kagglehub.dataset_download("ashery/chexpert")
# print("Path to dataset files:", path1)
# print("Path to dataset files:", path2)

# %%
from PIL import Image
from glob import glob
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from helpers import (
    plot_avg_update_to_weight_ratio_kde,
    plot_param_kde,
    roc_curves,
    confusion_matrices,
)


# %%
my_glob = glob("../data/CheXpert-v1.0-small/train/patient*/study*/*.jpg")
print("Number of Observations: ", len(my_glob))

# %%
train_df = pd.read_csv("../data/CheXpert-v1.0-small/train.csv")
print(f"the shape of the training dataset is : {train_df.shape}")
train_df.head()

train_df = train_df.sample(20000, replace=False)
print(f"the shape of the training dataset is : {train_df.shape}")

# %%
valid_df = pd.read_csv("../data/CheXpert-v1.0-small/valid.csv")
print(f"the shape of the validation dataset is : {valid_df.shape}")
valid_df.head()

my_glob_valid = glob("../data/CheXpert-v1.0-small/valid/patient*/study*/*.jpg")
print("Number of Observations: ", len(my_glob_valid))
print(f"the shape of the validation dataset is : {valid_df.shape}")

# %% [markdown]
# The targets have 3 possible values.
## 1 -> Positive
## 0 -> Negative
## -1 -> Uncertain
# There's several ways of handling these. We'll start off with the simplest which is to set the uncertain to negative.
# We'll explore other alternatives later on.
# We will also start with only Frontal images.


# %%
def prepare_dataset(dataframe, policy, class_names):
    dataset_df = dataframe[
        dataframe["Frontal/Lateral"] == "Frontal"
    ]  # take frontal pics only
    # Check which images had a support device. This is optional.
    # dataset_df = dataset_df[dataset_df["Support Devices"] == 1].reset_index(drop=True)
    df = dataset_df.sample(
        frac=1.0, random_state=1
    )  # If desired, downsample the dataset.
    df.fillna(0, inplace=True)  # fill the with zeros
    x_path, y_df = df["Path"].to_numpy(), df[class_names]
    class_ones = ["Atelectasis", "Cardiomegaly"]
    y = np.empty(y_df.shape, dtype=int)
    for i, (index, row) in enumerate(y_df.iterrows()):
        labels = []
        for cls in class_names:
            curr_val = row[cls]
            feat_val = 0
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1:
                    feat_val = 1
                elif curr_val == -1:
                    if policy == "ones":
                        feat_val = 1
                    elif policy == "zeroes":
                        feat_val = 0
                    elif policy == "mixed":
                        if cls in class_ones:
                            feat_val = 1
                        else:
                            feat_val = 0
                    else:
                        feat_val = 0
                else:
                    feat_val = 0
            else:
                feat_val = 0

            labels.append(feat_val)

        y[i] = labels

    x_path = ["../data/" + x for x in x_path]

    return x_path, y


# %%
# class_names = [
#     "Atelectasis",
#     "Cardiomegaly",
#     "Consolidation",
#     "Edema",
#     "Pleural Effusion",
# ]
# Reduce class names if necessary
class_names = [
    # "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    # "Lung Lesion",
    "Edema",
    # "Consolidation",
    "Pneumonia",
    "Atelectasis",
    # "Pneumothorax",
    "Pleural Effusion",
    # "Pleural Other",
    # "Fracture",
]
num_classes = len(class_names)

policy = ["ones", "zeroes", "mixed"]
policy = "ones"

x_path, labels = prepare_dataset(train_df, policy, class_names)
valid_x_path, valid_labels = prepare_dataset(valid_df, policy, class_names)

# %%
train_df = pd.DataFrame({"path": x_path})
df2 = pd.DataFrame(labels, columns=class_names)
train_df[list(df2.columns)] = df2

valid_df = pd.DataFrame({"path": valid_x_path})
df2_valid = pd.DataFrame(valid_labels, columns=class_names)
valid_df[list(df2.columns)] = df2_valid

print(f"Number of train datapoints: {len(train_df)}")
print(f"Number of valid datapoints: {len(valid_df)}")

# %%
# Plot the distribution of diseases
plt.figure()
plt.barh(df2.sum(axis=0).sort_values().index, df2.sum(axis=0).sort_values() / len(df2))
plt.xlabel("Frequency")
plt.ylabel("Disease")
plt.show()

# plt.figure()
# plt.barh(df2.sum(axis=0).sort_values().index, df2.sum(axis=0).sort_values())
# plt.xlabel("Count")
# plt.ylabel("Disease")
# plt.show()

# %%
plt.figure()
plt.barh(
    df2_valid.sum(axis=0).sort_values().index,
    df2_valid.sum(axis=0).sort_values() / len(df2_valid),
)
plt.xlabel("Frequency")
plt.ylabel("Disease")
plt.show()


# %%
class CheXpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Define separate transformations for RGB and grayscale images
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.grayscale_transform = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert grayscale to "RGB" with 3 identical channels
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]
                ),  # Same mean and std across channels
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]  # Assuming image paths are in the first column
        label = self.df.iloc[idx, 1:].values.astype(
            float
        )  # Multi-label format for each class

        # Open the image and check its mode
        image = Image.open(img_path)
        if image.mode == "L":  # Grayscale
            image = self.grayscale_transform(image)
        elif image.mode == "RGB":  # Color
            image = self.rgb_transform(image)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        return image, torch.tensor(label, dtype=torch.float32)


# %%
train_dataset = CheXpertDataset(train_df)
train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=20
)

valid_dataset = CheXpertDataset(valid_df)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=len(valid_df), shuffle=True, num_workers=20
)


# %%
class CustomConvNet(nn.Module):
    def __init__(self, num_classes=5, image_size=224):
        super(CustomConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        # Set dummy input to determine fully connected layer size
        self._set_fc_input_size(image_size)

        self.fc1 = nn.Linear(
            self.fc_input_size, 128
        )  # Adjust input size based on input image resolution
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(128, num_classes)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _set_fc_input_size(self, image_size):
        # Pass a dummy input through conv layers to dynamically calculate the size
        dummy_input = torch.zeros(1, 3, image_size, image_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        self.fc_input_size = x.view(-1).shape[0]  # Flatten and get size for fc1 input

    def forward(self, x):
        # Apply first conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Apply second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Apply third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)  # Output layer

        return x

    def _initialize_weights(self, layer):
        """Applies weight initialization based on the layer type."""
        if isinstance(layer, nn.Conv2d):
            init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0)


# Example usage
model = CustomConvNet(num_classes=num_classes)
print(model)

# for param in model.parameters():
#     param.requires_grad = True

# %%
### TODO: Evaluate different initializations and different networks.
# model = models.resnet50(weights="IMAGENET1K_V2")
# model = models.resnet152(weights="IMAGENET1K_V1")  # too big for this gpu
# model = models.vgg16(weights="IMAGENET1K_V1")
# model = models.densenet121(weights="DEFAULT")
model = models.efficientnet_b0(weights="DEFAULT")
# model = models.inception_v3(weights="IMAGENET1K_V1")
# model = models.mobilenet_v2(weights="IMAGENET1K_V2")

intermediate_size = 256  # Size of the new FC layer, adjust as needed
# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = True

# For VGG16
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# For EfficientNet or MobileNet
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# in_features = model.fc.in_features

# ResNet
# model.fc = nn.Sequential(
#     nn.Linear(in_features, intermediate_size),
#     nn.ReLU(),  # Add an activation function after the intermediate layer
#     nn.Linear(intermediate_size, num_classes),
# )

# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.fc.requires_grad = True

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
summary(model, input_size=(3, 224, 224))
# %%
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


# %%
def compute_update_to_weight_ratio(model, optimizer):
    """Calculates and returns the average update-to-weight ratio for each layer."""
    ratios = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Calculate weight update (gradient * learning rate)
            for group in optimizer.param_groups:
                lr = group["lr"]

            update = param.grad * lr
            # Calculate the update-to-weight ratio for the layer and average it
            update_to_weight_ratio = (
                (update / (param + 1e-8)).cpu().detach().numpy().flatten()
            )
            avg_ratio = update_to_weight_ratio.mean()
            ratios[name] = avg_ratio
    return ratios


# %%
def train_one_epoch():
    running_loss = 0.0
    running_f1 = 0.0

    for batch_idx, (images, labels) in enumerate(
        tqdm(train_dataloader, desc="Training", unit="batch")
    ):
        images, labels = images.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(images)
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Calculate accuracy for the batch
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        # batch_accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        batch_f1 = f1_score(
            labels.cpu(), predictions.cpu(), average="macro", zero_division=0
        )

        # # Update tqdm progress bar with batch loss and accuracy
        # tqdm.write(
        #     f"Batch {batch_idx + 1}/{len(train_dataloader)} - Training Loss: {loss.item():.4f}, "
        #     f"Training F1: {batch_f1 * 100:.2f}%"
        # )

        # Gather data and report
        running_loss += loss.item()
        running_f1 += batch_f1

        # Calculate and store update-to-weight ratio for each layer
        batch_layer_ratios = compute_update_to_weight_ratio(model, optimizer)
        for name, ratio in batch_layer_ratios.items():
            epoch_layer_ratios[name].append(ratio)

    running_loss /= batch_idx + 1
    running_f1 /= batch_idx + 1

    return running_loss, running_f1


# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0
EPOCHS = 10
layer_ratios_over_epochs = {
    name: [] for name, param in model.named_parameters() if param.requires_grad
}
train_loss_per_epoch = []
val_loss_per_epoch = []
train_f1_per_epoch = []
val_f1_per_epoch = []

start_time = time.time()

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))
    epoch_layer_ratios = {
        name: [] for name in layer_ratios_over_epochs
    }  # Track each batch's ratio for the epoch

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, running_f1 = train_one_epoch()

    train_loss_per_epoch.append(avg_loss)
    train_f1_per_epoch.append(running_f1)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    # Calculate the average update-to-weight ratio for each layer over the entire epoch
    for name in layer_ratios_over_epochs:
        avg_epoch_ratio = sum(epoch_layer_ratios[name]) / len(epoch_layer_ratios[name])
        layer_ratios_over_epochs[name].append(avg_epoch_ratio)
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vimgs, vlabels in valid_dataloader:
            vimgs, vlabels = vimgs.to(device), vlabels.to(device)
            voutputs = model(vimgs)
            # Calculate accuracy for the batch
            predictions = (torch.sigmoid(voutputs) > 0.5).float()
            # batch_accuracy = accuracy_score(labels.cpu(), predictions.cpu())
            val_f1 = f1_score(
                vlabels.cpu(), predictions.cpu(), average="macro", zero_division=0
            )
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / len(valid_dataloader)
    print(f"Loss train {avg_loss:.4f} --- Loss valid {avg_vloss:.4f}")
    print(f"F1 train {running_f1:.4f} --- F1 valid {val_f1:.4f}")
    print(f"Total time thus far: {(time.time()-start_time)/60:.2f} mins.....")
    print("-----------------------------------------------------------------")

    val_loss_per_epoch.append(avg_vloss.cpu())
    val_f1_per_epoch.append(val_f1)
    torch.cuda.empty_cache()
    epoch_number += 1

print(f"Total time to train: {(time.time()-start_time)/60:.2f} mins.....")

# %%
# PLot the train and val loss
plt.figure()
plt.title("Loss")
plt.plot(train_loss_per_epoch, label="train")
plt.plot(val_loss_per_epoch, label="val")
# plt.ylim((0, 0.01))
plt.legend()
plt.show()

# PLot the train and val F1
plt.figure()
plt.title("F1 score")
plt.plot(train_f1_per_epoch, label="train")
plt.plot(val_f1_per_epoch, label="val")
# plt.ylim((0, 0.01))
plt.legend()
plt.show()


# %%
# Plot update-to-weight ratio over epochs for each layer
plt.figure(figsize=(12, 8))
avg_ratio = np.array([0.0] * EPOCHS)
for name, ratios in layer_ratios_over_epochs.items():
    plt.plot(np.log10(np.abs(ratios)), label=name)
    avg_ratio += np.array(ratios)
plt.plot(np.log10(np.abs(avg_ratio / len(layer_ratios_over_epochs))), "k")
plt.title("Average Update-to-Weight Ratio Over Epochs (Each Layer)")
plt.xlabel("Epoch")
plt.ylabel("Average Update-to-Weight Ratio")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
plt.show()


# %%
# Diagnostics
# plot_avg_update_to_weight_ratio_kde(model, optimizer)
plot_param_kde(model)  # Plot parameter distributions at the start or periodically

# %%
# EVALS
model.eval()
# eval_on_train = "Train"
eval_on_train = "Valid"

# Disable gradient calculations for evaluation
valid_loss = 0.0
all_targets = []
all_outputs = []

# pick a dataloader
train_dataset = CheXpertDataset(train_df)
train_dataloader = DataLoader(
    train_dataset, batch_size=len(train_df), shuffle=True, num_workers=20
)

valid_dataset = CheXpertDataset(valid_df)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=len(valid_df), shuffle=True, num_workers=20
)

dataloader = train_dataloader if eval_on_train == "Train" else valid_dataloader


with torch.no_grad():
    for images, labels in dataloader:
        # Move images and labels to device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)
        # Forward pass to get logits
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        valid_loss += loss.item() * images.size(
            0
        )  # Accumulate loss scaled by batch size

        # Store predictions and true labels for metric calculation
        all_outputs.append(
            torch.sigmoid(outputs).cpu()
        )  # Convert logits to probabilities
        all_targets.append(labels.cpu())

# Calculate the average loss across the entire validation set
valid_loss /= len(dataloader.dataset)

# Concatenate all predictions and labels
all_outputs = torch.cat(all_outputs)
all_targets = torch.cat(all_targets)

# Example calculation of F1-score or accuracy
predictions = (all_outputs > 0.5).float()  # Convert probabilities to binary predictions
# accuracy = accuracy_score(all_targets, predictions)
f1 = f1_score(
    all_targets, predictions, average="macro"
)  # Use 'micro' or 'weighted' as needed

print(f"{eval_on_train} Loss: {valid_loss:.4f}")
print(f"{eval_on_train} F1 Score: {f1:.4f}")

# %%
roc_curves(class_names, all_outputs, all_targets)

# %%
confusion_matrices(class_names, all_outputs, all_targets)

# %%

# %%

# %% [markdown]
# In this script, we will run a large scale experiment to search for the best hyperparameters for our network.

# %%
from PIL import Image
from glob import glob
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import pandas as pd

import optuna
import mlflow

from helpers import prepare_dataset


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

def train_model(hparams):

    start_time = time.time()

    my_glob = glob("./CheXpert-v1.0-small/train/patient*/study*/*.jpg")
    print("Number of Observations: ", len(my_glob))
    train_df = pd.read_csv("./CheXpert-v1.0-small/train.csv")
    print(f"the shape of the training dataset is : {train_df.shape}")
    train_df.head()
    train_df = train_df.sample(1000, replace=False)
    print(f"the shape of the training dataset is : {train_df.shape}")
    valid_df = pd.read_csv("./CheXpert-v1.0-small/valid.csv")
    print(f"the shape of the validation dataset is : {valid_df.shape}")
    valid_df.head()
    my_glob_valid = glob("./CheXpert-v1.0-small/valid/patient*/study*/*.jpg")
    print("Number of Observations: ", len(my_glob_valid))
    print(f"the shape of the validation dataset is : {valid_df.shape}")

    class_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    num_classes = len(class_names)

    policy = ["ones", "zeroes", "mixed"]
    policy = "ones"

    x_path, labels = prepare_dataset(train_df, policy, class_names)
    valid_x_path, valid_labels = prepare_dataset(valid_df, policy, class_names)
    
    train_df = pd.DataFrame({"path": x_path})
    df2 = pd.DataFrame(labels, columns=class_names)
    train_df[list(df2.columns)] = df2

    valid_df = pd.DataFrame({"path": valid_x_path})
    df2_valid = pd.DataFrame(valid_labels, columns=class_names)
    valid_df[list(df2.columns)] = df2_valid

    print(f"Number of train datapoints: {len(train_df)}")
    print(f"Number of valid datapoints: {len(valid_df)}")

    train_dataset = CheXpertDataset(train_df)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=10
    )

    valid_dataset = CheXpertDataset(valid_df)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_df), shuffle=True, num_workers=10
    )

    # Define ANN
    model = models.resnet50(weights="DEFAULT")
    fc_size = 256
    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, fc_size),
        nn.ReLU(),  # Add an activation function after the intermediate layer
        nn.Linear(fc_size, num_classes),
    )
    model.fc.requires_grad = True
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_vloss = 0

    for epoch in range(hparams["epochs"]):
        print("EPOCH {}:".format(epoch + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        for images, labels in tqdm(train_dataloader, desc="Training", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            last_train_loss = loss.item()

        # Set the model to evaluation mode.
        model.eval()
        with torch.no_grad():
            for vimgs, vlabels in valid_dataloader:
                vimgs, vlabels = vimgs.to(device), vlabels.to(device)
                voutputs = model(vimgs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(valid_dataloader)
        print(f"Loss train {last_train_loss:.4f} --- Loss valid {avg_vloss:.4f}")
        print(f"Total time thus far: {(time.time()-start_time)/60:.2f} mins.....")
        print("-----------------------------------------------------------------")
        torch.cuda.empty_cache()

    print(f"Total time to train: {(time.time()-start_time)/60:.2f} mins.....")
    return avg_vloss, model


# %%
def objective(trial):
    # Suggest hyperparameters
    hparams = {
        # 'hidden_size': trial.suggest_int('hidden_size', 50, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'epochs': trial.suggest_int('epochs', 1, 5)
    }

    # Start an MLflow run
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(hparams)

        # Train the model
        avg_vloss, model = train_model(hparams)

        # Log the validation accuracy
        mlflow.log_metric('avg_vloss', avg_vloss)

        # Log the model
        mlflow.pytorch.log_model(model, 'model')

    return avg_vloss


# %%
if __name__ == '__main__':
    # Set up MLflow experiment
    mlflow.set_experiment('pytorch_hyperparameter_optimization')

    # Create an Optuna study
    study = optuna.create_study(direction='minimize')

    # Optimize the objective function
    study.optimize(objective, n_trials=5)

    # Print the best hyperparameters
    print('Best hyperparameters:', study.best_params)
    print('Best validation loss:', study.best_value)



# %%

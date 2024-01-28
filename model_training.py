import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import PneumoniaModel

# Defining a loader function, **load_file**, which defines how the files shall be loaded.
def load_file(path):
    return np.load(path).astype(np.float32)

train_transforms = transforms.Compose([
                                      transforms.ToTensor(),  # Converting numpy array to tensor
                                      transforms.Normalize(0.49, 0.248),  # Using mean and std from preprocessing notebook
                                      transforms.RandomAffine( # Data Augmentation
                                      degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                      transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
                                    ])

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
                                    ])

# Creating train and the val daatset corresponding data loaders
train_dataset = torchvision.datasets.DatasetFolder(
    "Data/Processed/train/",
    loader=load_file, extensions="npy", transform=train_transforms)

val_dataset = torchvision.datasets.DatasetFolder(
    "Data/Processed/val/",
    loader=load_file, extensions="npy", transform=val_transforms)
    
batch_size = 16
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")
print(f"Size of train loader: {len(train_loader)} and size test loader: {len(val_loader)}")

#Instanciating model
model = PneumoniaModel() 

# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
                                      monitor='Val Acc',
                                      save_top_k=10,
                                      mode='max'
                                      )

# Creating the trainer
gpus = 0
trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
                     callbacks=checkpoint_callback,
                     max_epochs=35
                     )

trainer.fit(model, train_loader, val_loader)

preds = []
labels = []

# MOdel Evaluation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Computing prediction on the entire validation set and storing predictions and labels
with torch.no_grad():
    for data, label in tqdm(val_dataset):
        data = data.to(device).float().unsqueeze(0)
        pred = torch.sigmoid(model(data)[0].cpu())
        preds.append(pred)
        labels.append(label)
preds = torch.tensor(preds)
labels = torch.tensor(labels).int()

'''Compute metrics:
We can see that the overall result is already decent with our simple model.<br />
However, we suffer from a large amount of False Negatives due to the data imbalance.<br />
This is of particular importance in to avoid in medical imaging as missing findings might be fatal.<br />'''

accuracy = torchmetrics.Accuracy()(preds, labels)
precision = torchmetrics.Precision()(preds, labels)
recall = torchmetrics.Recall()(preds, labels)
cm = torchmetrics.ConfusionMatrix(num_classes=2)(preds, labels)
cm_threshed = torchmetrics.ConfusionMatrix(num_classes=2, threshold=0.25)(preds, labels)

print(f"Validation Accuracy: {accuracy}")
print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")
print(f"Confusion Matrix:\n {cm}")
print(f"Confusion Matrix 2:\n {cm_threshed}")
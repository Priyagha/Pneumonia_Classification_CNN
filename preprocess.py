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
import pandas as pd
import pydicom
from pathlib import Path
import cv2

# We need the csv file containnig the labels
labels = pd.read_csv("/path/to/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
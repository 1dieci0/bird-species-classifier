import torch
import pandas as pd
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))


def load_classes(DATASET_PATH):
    classes = pd.read_csv(
        os.path.join(DATASET_PATH, "classes.txt"),
        sep=" ",
        names=["class_id", "class_name"]
    )
    return classes["class_name"].tolist()
  

def plot_accuracy():
    return

def plot_loss():
    return



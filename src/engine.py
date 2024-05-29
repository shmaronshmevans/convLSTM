import sys

sys.path.append("..")

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import gc

import model

from processing import create_data_for_convLSTM
from processing import read_in_images
from processing import get_time_title
from evaluate import save_output

import numpy as np


class ImageSequenceDataset(Dataset):
    def __init__(self, image_list, dataframe, target, sequence_length, transform=None):
        self.image_list = image_list
        self.dataframe = dataframe
        self.transform = transform
        self.sequence_length = sequence_length
        self.target = target

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        images = []
        i_start = max(0, i - self.sequence_length + 1)

        for j in range(i_start, i + 1):
            if j < len(self.image_list):
                img_name = self.image_list[j]
                image = np.load(img_name).astype(np.float32)
                image = image[:, :, 3:]
                if self.transform:
                    image = self.transform(image)
                images.append(torch.tensor(image))
            else:
                pad_image = torch.zeros_like(images[0])
                images.append(pad_image)

        while len(images) < self.sequence_length:
            pad_image = torch.zeros_like(images[0])
            images.insert(0, pad_image)

        images = torch.stack(images)
        images = images.to(torch.float32)

        # Extract target values
        y = self.dataframe[self.target].values[i_start : i + 1]
        if len(y) < self.sequence_length:
            pad_width = (self.sequence_length - len(y), 0)
            y = np.pad(y, (pad_width, (0, 0)), "constant", constant_values=0)

        y = torch.tensor(y).to(torch.float32)
        return images, y


def train_model(data_loader, model, optimizer, device, epoch, loss_func):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)
        # Forward pass and loss computation.
        output, last_states = model(X)
        # Squeeze unnecessary dimensions and transpose output tensor
        loss = loss_func(output[:, -1, :], y.squeeze()[:, -1, :])

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()
        gc.collect()

    # Synchronize and aggregate losses in distributed training.
    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches

    # Print the average loss on the master process (rank 0).
    print("epoch", epoch, "train_loss:", avg_loss)

    return avg_loss


def test_model(data_loader, model, device, epoch, loss_func):
    # Test a deep learning model on a given dataset and compute the test loss.
    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass to obtain model predictions.
        output, last_states = model(X)

        # Compute loss and add it to the total loss.
        total_loss += loss_func(output[:, -1, :], y.squeeze()[:, -1, :]).item()
        gc.collect()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def main(
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    CLIM_DIV,
    sequence_length,
    forecast_hour,
    num_layers,
    kernel_size,
    single,
):
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="convLSTM_beta",
        workspace="shmaronshmevans",
    )
    torch.manual_seed(101)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    today_date, today_date_hr = get_time_title.get_time_title(CLIM_DIV)
    # create data
    # df_train_ls, df_test_ls, features, stations = (
    #     create_data_for_convLSTM.create_data_for_model(
    #         CLIM_DIV, today_date, forecast_hour, single
    #     )
    # )

    train_df, test_df, train_ims, test_ims, target, stations = (
        read_in_images.create_data_for_model(CLIM_DIV)
    )

    # # load datasets
    # train_dataset = MultiStationDataset(
    #     df_train_ls, "target_error", features, sequence_length, forecast_hour
    # )
    # test_dataset = MultiStationDataset(
    #     df_test_ls, "target_error", features, sequence_length, forecast_hour
    # )

    train_dataset = ImageSequenceDataset(train_ims, train_df, target, sequence_length)
    test_dataset = ImageSequenceDataset(test_ims, test_df, target, sequence_length)

    # define model parameters
    ml = model.ConvLSTM(
        input_dim=int(26),
        hidden_dim=[26, 26],
        kernel_size=kernel_size,
        num_layers=num_layers,
        target=len(target),
        future_steps=forecast_hour,
    )
    if torch.cuda.is_available():
        ml.cuda()

    # Adam Optimizer
    optimizer = torch.optim.AdamW(ml.parameters(), lr=LEARNING_RATE)
    # MSE Loss
    loss_func = nn.MSELoss()
    # loss_func = FocalLossV3()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    hyper_params = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "clim_div": str(CLIM_DIV),
        "forecast_hour": forecast_hour,
        "num_layers": num_layers,
        "kernel_size": kernel_size,
    }
    # early_stopper = EarlyStopper(20)

    for ix_epoch in range(1, EPOCHS + 1):
        print("Epoch", ix_epoch)
        train_loss = train_model(
            train_loader, ml, optimizer, device, ix_epoch, loss_func
        )
        test_loss = test_model(test_loader, ml, device, ix_epoch, loss_func)
        print()
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        scheduler.step(test_loss)
        # if early_stopper.early_stop(test_loss):
        #     print(f"Early stopping at epoch {ix_epoch}")
        #     break

    save_output.eval_model(
        train_loader,
        test_loader,
        ml,
        device,
        target,
        train_df,
        test_df,
        stations,
        today_date,
        today_date_hr,
        CLIM_DIV,
        forecast_hour,
        sequence_length,
    )
    experiment.end()


main(
    EPOCHS=100,
    BATCH_SIZE=int(48),
    LEARNING_RATE=7e-5,
    CLIM_DIV="Mohawk Valley",
    sequence_length=12,
    forecast_hour=4,
    num_layers=2,
    kernel_size=(3, 3),
    single=False,
)

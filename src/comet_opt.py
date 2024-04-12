import sys

sys.path.append("..")

import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model
from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model
from comet_ml import Optimizer
import gc

import model

from processing import create_data_for_convLSTM
from processing import get_time_title
from evaluate import save_output


class MultiStationDataset(Dataset):
    def __init__(
        self, dataframes, target, features, sequence_length, forecast_hour, nysm_vars=14
    ):
        """
        dataframes: list of station dataframes like in the SequenceDataset
        target: target error
        features: list of features for model
        sequence_length: int
        """
        self.dataframes = dataframes
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_hour = forecast_hour
        self.nysm_vars = nysm_vars

    def __len__(self):
        shaper = min(
            [
                self.dataframes[i].values.shape[0] - (self.sequence_length)
                for i in range(len(self.dataframes))
            ]
        )
        return shaper

    def __getitem__(self, i):
        # this is the preceeding sequence_length timesteps
        x = torch.stack(
            [
                torch.tensor(
                    dataframe[self.features].values[i : (i + self.sequence_length)]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)

        # stacking the sequences from each dataframe along a new axis, so the output is of shape (batch, stations (len(self.dataframes)), past_steps, features)
        y = torch.stack(
            [
                torch.tensor(
                    dataframe[self.target].values[i : i + self.sequence_length]
                )
                for dataframe in self.dataframes
            ]
        ).to(torch.float32)
        # this is (batch, stations, future_steps)
        x[-self.forecast_hour :, : self.nysm_vars] = x[
            self.forecast_hour, : self.nysm_vars
        ]  # check that this is setting the right positions to this value
        # Transpose x to have dimensions [sequence_length, features, stations]
        x = x.permute(1, 2, 0)
        return x, y


def train_model(data_loader, model, optimizer, device, epoch, loss_func):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.unsqueeze(3)
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output, last_states = model(X)
        # Squeeze unnecessary dimensions and transpose output tensor
        loss = loss_func(output, y)

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
        X = X.unsqueeze(3)
        X, y = X.to(device), y.to(device)
        # Forward pass to obtain model predictions.
        output, last_states = model(X)

        # Compute loss and add it to the total loss.
        total_loss += loss_func(output, y).item()
        gc.collect()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def main(
    LEARNING_RATE,
    sequence_length,
    num_layers,
    kernel_size,
    forecast_hour=4,
    EPOCHS=10,
    BATCH_SIZE=int(100),
    CLIM_DIV="Mohawk Valley",
):
    torch.manual_seed(101)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    today_date, today_date_hr = get_time_title.get_time_title(CLIM_DIV)
    # create data
    df_train_ls, df_test_ls, features, stations = (
        create_data_for_convLSTM.create_data_for_model(
            CLIM_DIV, today_date, forecast_hour
        )
    )

    # load datasets
    train_dataset = MultiStationDataset(
        df_train_ls, "target_error", features, sequence_length, forecast_hour
    )
    test_dataset = MultiStationDataset(
        df_test_ls, "target_error", features, sequence_length, forecast_hour
    )

    kernel = (kernel_size, kernel_size)
    print(kernel)
    # define model parameters
    ml = model.ConvLSTM(
        input_dim=len(features),
        hidden_dim=len(features),
        kernel_size=kernel,
        num_layers=num_layers,
    )
    if torch.cuda.is_available():
        ml.cuda()

    # Adam Optimizer
    optimizer = torch.optim.Adam(ml.parameters(), lr=LEARNING_RATE)
    # MSE Loss
    loss_func = nn.MSELoss()

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
        "kernel_size": kernel,
    }

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

    return test_loss


config = {
    # Pick the Bayes algorithm:
    "algorithm": "bayes",
    # Declare what to optimize, and how:
    "spec": {
        "metric": "loss",
        "objective": "minimize",
    },
    # Declare your hyperparameters:
    "parameters": {
        "num_layers": {"type": "integer", "min": 1, "max": 20},
        "kernel_size": {"type": "integer", "min": 1, "max": 20},
        "sequence_length": {"type": "integer", "min": 36, "max": 500},
        "learning_rate": {"type": "float", "min": 5e-20, "max": 1e-3},
    },
    "trials": 30,
}

print("!!! begin optimizer !!!")

opt = Optimizer(config)

# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(
    project_name="hyperparameter-tuning-for-convLSTM"
):
    loss = main(
        LEARNING_RATE=experiment.get_parameter("learning_rate"),
        sequence_length=experiment.get_parameter("sequence_length"),
        num_layers=experiment.get_parameter("num_layers"),
        kernel_size=experiment.get_parameter("kernel_size"),
    )

    experiment.log_metric("loss", loss)
    experiment.end()

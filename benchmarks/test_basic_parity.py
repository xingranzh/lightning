import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, MNIST

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.seed import seed_everything

_PATH_DATASETS = "~/data/"


class AverageDataset(Dataset):
    def __init__(self, dataset_len=300, sequence_len=100):
        self.dataset_len = dataset_len
        self.sequence_len = sequence_len
        self.input_seq = torch.randn(dataset_len, sequence_len, 10)
        top, bottom = self.input_seq.chunk(2, -1)
        self.output_seq = top + bottom.roll(shifts=1, dims=-1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item]


class ParityModuleRNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.linear_out = nn.Linear(in_features=20, out_features=5)
        self.example_input_array = torch.rand(2, 3, 10)

    def forward(self, x):
        seq, last = self.rnn(x)
        return self.linear_out(seq)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(AverageDataset(), batch_size=30)


class ParityModuleMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        self.c_d1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.3)
        self.c_d2 = nn.Linear(in_features=128, out_features=10)
        self.example_input_array = torch.rand(2, 1, 28, 28)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(MNIST(root=_PATH_DATASETS, train=True, download=True), batch_size=128, num_workers=1)


class ParityModuleCIFAR(LightningModule):
    def __init__(self, backbone="resnet101", hidden_dim=1024, learning_rate=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = 10
        self.backbone = getattr(models, backbone)(pretrained=pretrained)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, hidden_dim), torch.nn.Linear(hidden_dim, self.num_classes)
        )
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            CIFAR10(root=_PATH_DATASETS, train=True, download=True, transform=self.transform),
            batch_size=32,
            num_workers=1,
        )


def assert_parity_relative(pl_values, pt_values, norm_by: float = 1, max_diff: float = 0.1):
    # assert speeds
    diffs = np.asarray(pl_values) - np.mean(pt_values)
    # norm by vanilla time
    diffs = diffs / norm_by
    # relative to mean reference value
    diffs = diffs / np.mean(pt_values)
    assert np.mean(diffs) < max_diff, f"Lightning diff {diffs} was worse than vanilla PT (threshold {max_diff})"


def assert_parity_absolute(pl_values, pt_values, norm_by: float = 1, max_diff: float = 0.55):
    # assert speeds
    diffs = np.asarray(pl_values) - np.mean(pt_values)
    # norm by event count
    diffs = diffs / norm_by
    assert np.mean(diffs) < max_diff, f"Lightning {diffs} was worse than vanilla PT (threshold {max_diff})"


def _hook_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = np.nan
    return used_memory


def measure_loops(cls_model: LightningModule, kind: str, num_epochs: int = 10):
    """Returns an array with the last loss from each epoch for each run.

    Args:
        cls_model: LightingModule to benchmark with.
        kind: "PT Lightning" or "Vanilla PT".
        num_epochs: number of epochs.
    """
    hist_losses = []
    hist_durations = []
    hist_memory = []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True

    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(1)

    time_start = time.perf_counter()

    _loop = lightning_loop if kind == "PT Lightning" else vanilla_loop
    final_loss, used_memory = _loop(cls_model, device_type=device_type, num_epochs=num_epochs)

    time_end = time.perf_counter()

    hist_losses.append(final_loss)
    hist_durations.append(time_end - time_start)
    hist_memory.append(used_memory)

    return {"losses": hist_losses, "durations": hist_durations, "memory": hist_memory}


def vanilla_loop(cls_model, device_type: str = "cuda", num_epochs=10):
    device = torch.device(device_type)
    seed_everything(123)
    model = cls_model()
    dl = model.train_dataloader()
    optimizer = model.configure_optimizers()
    model = model.to(device)
    epoch_losses = []
    for epoch in range(num_epochs):
        # run through full training set
        for j, batch in enumerate(dl):
            batch = [x.to(device) for x in batch]
            loss_dict = model.training_step(batch, j)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # track last epoch loss
        epoch_losses.append(loss.item())
    return epoch_losses[-1], _hook_memory()


def lightning_loop(cls_model, device_type: str = "cuda", num_epochs=10):
    seed_everything(123)
    torch.backends.cudnn.deterministic = True
    model = cls_model()
    trainer = Trainer(
        # as the first run is skipped, no need to run it long
        max_epochs=num_epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="gpu" if device_type == "cuda" else "cpu",
        devices=1,
        logger=False,
        replace_sampler_ddp=False,
    )
    trainer.fit(model)
    return trainer.fit_loop.running_loss.last().item(), _hook_memory()


class PyTorchParitySuite:
    def setup(self):
        if not torch.cuda.is_available():
            # skip this benchmark
            raise NotImplementedError

        # prepare datasets in setup

    # (ParityModuleRNN, 0.05, 0.001, 4, 3),
    # ParityModuleMNIST, 0.3, 0.001, 4, 3),  # todo: lower this thr
    # pytest.param(ParityModuleCIFAR, 4.0, 0.0002, 2, 2, marks=_MARK_SHORT_BM),
    param_names = ["cls_model", "num_epochs"]
    params = (ParityModuleMNIST, 4)

    def time_lightning(
        self,
        cls_model: LightningModule,
        num_epochs: int,
    ):
        measure_loops(cls_model, kind="PT Lightning", num_epochs=num_epochs)

    def time_pytorch(
        self,
        cls_model: LightningModule,
        num_epochs: int,
    ):
        measure_loops(cls_model, kind="Vanilla PT", num_epochs=num_epochs)
        # max_diff_speed: float,
        # max_diff_memory: float,
        # make sure the losses match exactly  to 5 decimal places
        # print(f"Losses are for... \n vanilla: {vanilla['losses']} \n lightning: {lightning['losses']}")
        # for pl_out, pt_out in zip(lightning["losses"], vanilla["losses"]):
        #     np.testing.assert_almost_equal(pl_out, pt_out, 5)
        # drop the first run for initialize dataset (download & filter)
        # assert_parity_absolute(
        #     lightning["durations"][1:], vanilla["durations"][1:], norm_by=num_epochs, max_diff=max_diff_speed
        # )
        # assert_parity_relative(lightning["memory"], vanilla["memory"], max_diff=max_diff_memory)

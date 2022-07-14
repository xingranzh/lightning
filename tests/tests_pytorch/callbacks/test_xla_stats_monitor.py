# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import XLAStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@RunIf(tpu=True)
def test_xla_stats_monitor(tmpdir):
    """Test XLA stats are logged using a logger."""

    model = BoringModel()
    with pytest.deprecated_call(match="The `XLAStatsMonitor` callback was deprecated in v1.5"):
        xla_stats = XLAStatsMonitor()

    class DebugLogger(CSVLogger):
        @rank_zero_only
        def log_metrics(self, metrics, step=None) -> None:
            fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]
            for f in fields:
                assert any(f in h for h in metrics.keys())

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=5,
        accelerator="tpu",
        devices=8,
        callbacks=[xla_stats],
        logger=DebugLogger(tmpdir),
    )
    trainer.fit(model)


@RunIf(tpu=True)
def test_xla_stats_monitor_no_logger(tmpdir):
    """Test XLAStatsMonitor with no logger in Trainer."""

    model = BoringModel()
    with pytest.deprecated_call(match="The `XLAStatsMonitor` callback was deprecated in v1.5"):
        xla_stats = XLAStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir, callbacks=[xla_stats], max_epochs=1, accelerator="tpu", devices=[1], logger=False
    )

    with pytest.raises(MisconfigurationException, match="Trainer that has no logger."):
        trainer.fit(model)


@RunIf(tpu=True)
def test_xla_stats_monitor_no_tpu_warning(tmpdir):
    """Test XLAStatsMonitor raises a warning when not training on TPUs."""

    model = BoringModel()
    with pytest.deprecated_call(match="The `XLAStatsMonitor` callback was deprecated in v1.5"):
        xla_stats = XLAStatsMonitor()

    trainer = Trainer(default_root_dir=tmpdir, callbacks=[xla_stats], max_steps=1, tpu_cores=None)

    with pytest.raises(MisconfigurationException, match="not running on TPU"):
        trainer.fit(model)

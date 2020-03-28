import argparse
import os
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from cifar10 import Module, utils

warnings.filterwarnings("ignore")

utils.set_seed(0)


def main(hparams):
    engine = Module(hparams)

    mlf_logger = MLFlowLogger(experiment_name=hparams.exp_name, tracking_uri="./mlruns")

    exp = mlf_logger.experiment.get_experiment_by_name(hparams.exp_name)
    artifacts_dir = os.path.join(exp.artifact_location, mlf_logger.run_id, "artifacts")

    checkpoint_callback = ModelCheckpoint(
        filepath=artifacts_dir, save_top_k=-1, verbose=True, monitor="val_loss_avg", mode="min", prefix="",
    )

    trainer = Trainer(
        logger=mlf_logger, checkpoint_callback=checkpoint_callback, max_epochs=hparams.num_epochs, gpus=[0]
    )
    trainer.fit(engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp-name", default="Default")
    parser.add_argument("--num-epochs", default=128, type=int)

    parser = Module.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)

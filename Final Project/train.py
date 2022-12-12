from fastai.vision.all import *
from fastai.callback.all import *
from fastai.callback.wandb import *

import wandb
import logging
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp

from utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Hellper functions
def label_func(path, fname):
    dataset = fname.parent.parent.name
    return (
        path
        / f"gtFine_trainvaltest/gtFine/{dataset}"
        / fname.parent.name
        / f"{fname.name.rsplit('_', 1)[0]}_gtFine_labelIds.png"
    )


def get_dls(path, bs):
    codes = np.loadtxt(path / "city_codes.txt", dtype=str)
    cityscape = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes)),
        get_items=get_image_files,
        get_y=partial(label_func, path),
        splitter=GrandparentSplitter(valid_name="val"),
        batch_tfms=aug_transforms(size=(128, 256)),
    )
    dls = cityscape.dataloaders(
        path / f"leftImg8bit_trainvaltest/leftImg8bit/", path=path, bs=bs
    )
    return dls


def train(conf):
    # data
    path = Path("/mnt/e/Downloads/City Scapes dataset/")
    dls = get_dls(path, conf.bs)

    # model

    arch = load_obj(conf.arch.replace("smp", "segmentation_models_pytorch"))
    cbs = [
        WandbCallback(log_model=False, model_name=arch.__name__),
        GradientAccumulation(n_acc=2),
    ]

    # training
    wandb.init(project=conf.project, name=f"{arch.__name__}_{conf.encoder}")
    learn = seg_learner(dls, arch, conf.encoder, "imagenet", cbs=cbs)
    learn.fine_tune(conf.epochs, freeze_epochs=2)

    # logging
    logger.info(f"--- Architecture : {arch.__name__} Encoder : {conf.encoder} ---")
    trn_loss, val_loss = learn.recorder.values[-1]
    logging.info(f"Training Loss : {trn_loss:.4f} and Validation Loss : {val_loss:.4f}")
    m_size = get_model_size(learn.model)
    logging.info(f"Model size: {m_size:.3f}MB")
    flops, params = get_thop(learn.model)
    logging.info(f"Parameters: {params}")
    logging.info(f"Flops: {flops}")

    wandb.finish()


if __name__ == "__main__":
    base_conf = OmegaConf.create(
        {
            "encoder": "mit_b2",
            "arch": "smp.MAnet",
            "epochs": 5,
            "project": "255_final_project",
            "bs": 8,
        }
    )
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_conf, cli_conf)
    logging.info(f" Using Parameters : {conf}")
    train(conf)

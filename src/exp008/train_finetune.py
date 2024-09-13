import argparse
import datetime
import io
import math
import os
import warnings
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple, Union

import albumentations as albu
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    callbacks,
    seed_everything,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.metrics import auc, roc_auc_score, roc_curve
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "008"
COMMENT = """
baseline, add roc-star loss, GeM, fix augmentation, input_bn
"""


# ref: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
def get_transforms(train: bool = False, image_size: int = 256) -> Callable:
    if train:
        return albu.Compose(
            [
                albu.Resize(image_size, image_size),
                albu.Transpose(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.75
                ),
                albu.OneOf(
                    [
                        albu.OpticalDistortion(),
                        albu.GridDistortion(),
                        albu.ElasticTransform(),
                    ],
                    p=0.7,
                ),
                albu.CLAHE(clip_limit=4.0, p=0.7),
                albu.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
                ),
                albu.ShiftScaleRotate(
                    border_mode=0,
                    p=0.85,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return albu.Compose(
            [
                albu.Resize(image_size, image_size),
                ToTensorV2(),
            ]
        )


def load_image(image_id: str, data: h5py.File) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(np.array(data[image_id]))))


def get_table_feat(
    meta_data: pd.DataFrame,
    idx: int,
) -> Union[Dict, Dict]:
    # TODO: feature engineering
    # cont_feats = [d[0] for d in meta_data.dtypes.to_dict().items() if d[1] == "object"]
    # cat_feats = [d[0] for d in meta_data.dtypes.to_dict().items() if d[1] == "float"]
    return {}


class ISIC2024Dataset(Dataset):
    def __init__(
        self, meta_data: pd.DataFrame, data: h5py.File, mode="train", image_size=256
    ):
        self.meta_data = meta_data
        self.data = data
        self.transforms = get_transforms(train=mode == "train", image_size=image_size)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.meta_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.meta_data.iloc[idx].isic_id
        image = load_image(image_id, self.data)
        image = self.transforms(image=image)["image"] / 255.0
        data = {}
        data["image"] = image
        data.update(get_table_feat(self.meta_data, idx))
        if self.mode != "test":
            data["target"] = torch.tensor(
                [self.meta_data.iloc[idx].target], dtype=torch.float32
            )
        return data


class ISIC2024DataModule(LightningDataModule):
    def __init__(
        self,
        meta_data: pd.DataFrame,
        data: h5py.File,
        fold: int = None,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.meta_data = meta_data
        self.data = data
        self.save_hyperparameters(ignore=["meta_data", "data"])

    def create_dataset(self, mode: str) -> Dataset:
        if self.hparams.fold is not None:
            if mode == "train":
                return ISIC2024Dataset(
                    self.meta_data[self.meta_data.fold_id != self.hparams.fold],
                    self.data,
                    mode=mode,
                    image_size=self.hparams.image_size,
                )
            else:
                return ISIC2024Dataset(
                    self.meta_data[self.meta_data.fold_id == self.hparams.fold],
                    self.data,
                    mode=mode,
                    image_size=self.hparams.image_size,
                )
        else:  # for inference
            assert mode == "test"
            return ISIC2024Dataset(
                self.meta_data,
                self.data,
                mode=mode,
                image_size=self.hparams.image_size,
            )

    def __dataloader(self, mode: str) -> DataLoader:
        return DataLoader(
            self.create_dataset(mode),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=(mode == "train"),
            drop_last=(mode == "train"),
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader("test")

    @staticmethod
    def add_argparse_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ISIC2024DataModule")
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=24)
        return parent_parser


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def gem(x, p=3, eps=1e-4):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-4, trainable=True):
        super(GeM, self).__init__()
        self.trainable = trainable
        self.p = nn.Parameter(torch.ones(1) * p) if self.trainable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[0], -1)


class ISIC2024Model2D(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_class: int = 1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(3)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            drop_path_rate=drop_path_rate,
            img_size=(
                128
                if ("swin" in model_name)
                or ("coat" in model_name)
                or ("max" in model_name)
                else None
            ),
        )
        self.output_fmt = getattr(self.encoder, "output_fmt", "NHCW")
        num_features = self.encoder.feature_info.channels()
        self.neck = nn.Sequential(
            nn.Conv2d(num_features[-1], 512, kernel_size=1),
            GeM(trainable=True),
            # AdaptiveConcatPool2d(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5, inplace=False),
            nn.Linear(512, num_class),
        )

    def forward_image_feats(self, img: torch.Tensor) -> torch.Tensor:
        # img -> (bs, c, h, w)
        img = self.input_bn(img)
        img_feats = self.encoder(img)
        if self.output_fmt == "NHWC":
            img_feats = [
                img_feat.permute(0, 3, 1, 2).contiguous() for img_feat in img_feats
            ]
        img_feat = img_feats[-1]
        img_feat = F.normalize(self.neck(img_feat), dim=1)
        return img_feat

    def forward_head(self, img_feat: torch.Tensor) -> torch.Tensor:
        output = self.head(img_feat)
        return output

    def forward(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """
        img: (bs, ch, h, w)
        """
        img_feat = self.forward_image_feats(img)
        return self.forward_head(img_feat)


class Mixup(object):
    def __init__(self, p: float = 0.5, alpha: float = 0.5):
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self):
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False
        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self):
        self.lam = 1.0


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(
        self, loss_fcn: nn.Module, gamma: float = 1.5, alpha: float = 0.25, smooth=0.0
    ):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.smooth = smooth
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        loss = self.loss_fcn(pred, true * (1 - (self.smooth / 0.5)) + self.smooth)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class RocStarLoss(_Loss):
    """Smooth approximation for ROC AUC"""

    def __init__(
        self,
        delta=1.0,
        sample_size=10000,
        sample_size_gamma=10000,
        update_gamma_each=50,
    ):
        r"""
        Args:
            delta: Param from article
            sample_size (int): Number of examples to take for ROC AUC approximation
            sample_size_gamma (int): Number of examples to take for Gamma parameter approximation
            update_gamma_each (int): Number of steps after which to recompute gamma value.
        """
        super().__init__()
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_gamma = sample_size_gamma
        self.update_gamma_each = update_gamma_each
        self.steps = 0
        size = max(sample_size, sample_size_gamma)

        # Randomly init labels
        self.y_pred_history = torch.rand((size, 1), device="cuda")
        self.y_true_history = torch.randint(2, (size, 1), device="cuda")

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Tensor of model predictions before using sigmoid. Shape (B x 1)
            y_true: Tensor of true labels in {0, 1}. Shape (B x 1)
        """
        y_pred = torch.sigmoid(y_pred)
        if self.steps % self.update_gamma_each == 0:
            self.update_gamma()
        self.steps += 1

        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]

        # Take last `sample_size` elements from history
        y_pred_history = self.y_pred_history[-self.sample_size :]
        y_true_history = self.y_true_history[-self.sample_size :]

        positive_history = y_pred_history[y_true_history > 0]
        negative_history = y_pred_history[y_true_history < 1]

        if positive.size(0) > 0:
            diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
            loss_positive = (torch.nn.functional.relu(diff) ** 2).mean()
        else:
            loss_positive = 0

        if negative.size(0) > 0:
            diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
            loss_negative = (torch.nn.functional.relu(diff) ** 2).mean()
        else:
            loss_negative = 0

        loss = loss_negative + loss_positive

        # Update FIFO queue
        batch_size = y_pred.size(0)
        self.y_pred_history = torch.cat(
            (self.y_pred_history[batch_size:], y_pred.clone().detach())
        )
        self.y_true_history = torch.cat(
            (self.y_true_history[batch_size:], y_true.clone().detach())
        )
        return loss

    def update_gamma(self):
        # Take last `sample_size_gamma` elements from history
        y_pred = self.y_pred_history[-self.sample_size_gamma :]
        y_true = self.y_true_history[-self.sample_size_gamma :]

        positive = y_pred[y_true > 0]
        negative = y_pred[y_true < 1]

        # Create matrix of size sample_size_gamma x sample_size_gamma
        diff = positive.view(-1, 1) - negative.view(1, -1)
        AUC = (diff > 0).type(torch.float).mean()
        num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)

        # Adjuct gamma, so that among correct ordered samples `delta * num_wrong_ordered` were considered
        # ordered incorrectly with gamma added
        correct_ordered = diff[diff > 0].flatten().sort().values
        if len(correct_ordered) != 0:
            idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered) - 1)
            self.gamma = correct_ordered[idx]
            # print(f"Updated gamma: {self.gamma}")
        # else:
        # print(f"Did not update gamma. Current gamma is: {self.gamma}")


# https://www.kaggle.com/code/metric/isic-pauc-abovetpr
def comp_metric(gt: np.ndarray, preds: np.ndarray, min_tpr: float = 0.80):
    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(gt - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * preds

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    return partial_auc


class ICIC2024LightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnetrs50",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        num_class: int = 1,
        mixup_p: float = 0.0,
        mixup_alpha: float = 0.5,
        no_mixup_epochs: int = 0,
        label_smoothing: float = 0.0,
        pos_weight: float = 10.0,
        lr: float = 1e-3,
        backbone_lr: float = None,
        weight_decay: float = 1e-4,
        precision: str = "32-true",
    ) -> None:
        super().__init__()
        # self.lr = lr
        # self.backbone_lr = backbone_lr if backbone_lr is not None else lr
        self.save_hyperparameters()
        self.__build_model(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_class=num_class,
        )
        self.mixupper = Mixup(p=mixup_p, alpha=mixup_alpha)
        self.gt_val = []
        self.logit_val = []
        self.hparams.backbone_lr = (
            self.hparams.backbone_lr if self.hparams.backbone_lr is not None else lr
        )

    def __build_model(
        self,
        model_name: str = "resnetrs50",
        pretrained: bool = False,
        drop_path_rate: float = 0,
        num_class: int = 2,
    ):
        self.model = ISIC2024Model2D(
            model_name=model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_class=num_class,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.99)
        self.criterions = {
            "bce": QFocalLoss(
                nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.hparams.pos_weight]))
            ),
            "rocstar": RocStarLoss(
                delta=1,
                sample_size=10000,
                sample_size_gamma=10000,
                update_gamma_each=50,
            ),
        }

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        target = labels["targets"].to(dtype=outputs["cls_logits"].dtype)
        if self.training and self.hparams.label_smoothing > 0:
            target = (
                target * (1 - (self.hparams.label_smoothing / 0.5))
                + self.hparams.label_smoothing
            )
        self.criterions["bce"].loss_fcn.pos_weight = self.criterions[
            "bce"
        ].loss_fcn.pos_weight.to(outputs["cls_logits"].device)
        losses["bce"] = self.criterions["bce"](
            outputs["cls_logits"],
            target,
        )
        losses["loss"] = losses["bce"]
        if self.training:
            losses["rocstar"] = self.criterions["rocstar"](
                outputs["cls_logits"], target
            )
            losses["loss"] += losses["rocstar"]
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        self.mixupper.init_lambda()
        if (
            self.mixupper.do_mixup
            and self.current_epoch
            < self.trainer.max_epochs - self.hparams.no_mixup_epochs
        ):
            batch["image"] = self.mixupper.lam * batch["image"] + (
                1 - self.mixupper.lam
            ) * batch["image"].flip(0)
        outputs["cls_logits"] = self.model(batch["image"])

        loss_target["targets"] = batch["target"]
        losses = self.calc_loss(outputs, loss_target)

        if (
            self.mixupper.do_mixup
            and self.current_epoch
            < self.trainer.max_epochs - self.hparams.no_mixup_epochs
        ):
            loss_target["targets"] = loss_target["targets"].flip(0)
            losses_b = self.calc_loss(outputs, loss_target)
            for key in losses:
                losses[key] = (
                    self.mixupper.lam * losses[key]
                    + (1 - self.mixupper.lam) * losses_b[key]
                )
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_bce_loss=losses["bce"],
                train_rocstar_loss=losses["rocstar"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}
        outputs["cls_logits"] = self.model_ema.module(batch["image"])

        loss_target["targets"] = batch["target"]
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.logit_val.append(
            torch.sigmoid(outputs["cls_logits"]).detach().cpu().numpy()
        )
        self.gt_val.append(batch["target"].detach().cpu().numpy() > 0.5)

        self.log_dict(
            dict(
                val_loss=losses["loss"],
            )
        )
        return step_output

    def on_validation_epoch_end(self):
        logit_val = np.concatenate(self.logit_val)[:, 0]  # (bs, cls)
        gt_val = np.concatenate(self.gt_val)[:, 0]  # (bs, cls)

        auc = roc_auc_score(gt_val, logit_val)
        pauc = comp_metric(gt_val, logit_val)

        self.logit_val.clear()
        self.gt_val.clear()

        self.log_dict(
            dict(
                val_auc=auc,
                val_pauc=pauc,
            ),
            sync_dist=True,
        )

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in list(self.model.encoder.named_parameters())
                    + list(self.model.input_bn.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.encoder.named_parameters())
                    + list(self.model.input_bn.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.backbone_lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.neck.named_parameters())
                    + list(self.model.head.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.lr,
            },
            {
                "params": [
                    p
                    for n, p in list(self.model.neck.named_parameters())
                    + list(self.model.head.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(
            self.get_optimizer_parameters(),
            eps=1e-5 if self.hparams.precision == "16-true" else 1e-8,
        )
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ICIC2024LightningModel")
        parser.add_argument(
            "--model_name",
            default="resnetrs50",
            type=str,
            metavar="MN",
            help="Name (as in ``timm``) of the feature extractor",
            dest="model_name",
        )
        parser.add_argument(
            "--drop_path_rate",
            default=None,
            type=float,
            metavar="DPR",
            dest="drop_path_rate",
        )
        parser.add_argument(
            "--mixup_p", default=0.0, type=float, metavar="MP", dest="mixup_p"
        )
        parser.add_argument(
            "--mixup_alpha", default=0.0, type=float, metavar="MA", dest="mixup_alpha"
        )
        parser.add_argument(
            "--no_mixup_epochs",
            default=0,
            type=int,
            metavar="NME",
            dest="no_mixup_epochs",
        )
        parser.add_argument(
            "--label_smoothing",
            default=0.0,
            type=float,
            metavar="LS",
            help="label smoothing",
            dest="label_smoothing",
        )
        parser.add_argument(
            "--pos_weight",
            default=10.0,
            type=float,
            metavar="PW",
            help="positive weight",
            dest="pos_weight",
        )
        parser.add_argument(
            "--lr",
            default=1e-3,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--backbone_lr",
            default=None,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="backbone_lr",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-4,
            type=float,
            metavar="WD",
            help="weight decay",
            dest="weight_decay",
        )
        return parent_parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--resume_dir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=1, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="precision setting",
    )
    parent_parser.add_argument(
        "--epochs", default=10, type=int, metavar="N", help="total number of epochs"
    )
    parent_parser.add_argument(
        "--fold", default=0, type=int, metavar="N", help="fold number"
    )
    parent_parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        metavar="agb",
        dest="accumulate_grad_batches",
    )
    parent_parser.add_argument(
        "--pos_upsample_rate",
        default=1,
        type=int,
        metavar="PUR",
        help="upsampling rate for positive sample",
        dest="pos_upsample_rate",
    )
    parent_parser.add_argument(
        "--neg_downsample_rate",
        default=1,
        type=int,
        metavar="NDR",
        help="downsampling rate for negative sample",
        dest="neg_downsample_rate",
    )
    parser = ICIC2024LightningModel.add_model_specific_args(parent_parser)
    parser = ISIC2024DataModule.add_argparse_args(parser)

    return parser.parse_args()


def main(args):
    seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    train_meta_data = pd.read_csv("../../input/isic-2024-challenge/train-metadata.csv")
    fold = pd.read_csv("../../input/isic-2024-challenge/fold.csv")
    train_meta_data = train_meta_data.merge(fold, on="isic_id", how="inner")
    train_images = h5py.File("../../input/isic-2024-challenge/train-image.hdf5", "r")
    assert args.fold < 5
    for fold in range(5):
        if args.fold != fold:
            continue
        train_meta_data = pd.concat(
            [train_meta_data]
            + [
                train_meta_data[
                    (train_meta_data.fold_id != fold) & (train_meta_data.target == 1)
                ]
                for _ in range(args.pos_upsample_rate - 1)
            ]
        )
        # negative downsample
        train_meta_data = pd.concat(
            [train_meta_data[train_meta_data.fold_id == fold]]
            + [
                train_meta_data[
                    (train_meta_data.fold_id != fold) & (train_meta_data.target == 1)
                ]
            ]
            + [
                train_meta_data[
                    (train_meta_data.fold_id != fold) & (train_meta_data.target == 0)
                ].sample(
                    frac=1 / args.neg_downsample_rate,
                    random_state=args.seed,
                )
            ]
        ).reset_index(drop=True)
        datamodule = ISIC2024DataModule(
            meta_data=train_meta_data,
            data=train_images,
            fold=fold,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        resume_checkpoint = glob(
            f"../../logs/exp{EXP_ID}/{args.resume_dir}/**/best_auc.ckpt", recursive=True
        )[0]
        print(f"load checkpoint: {resume_checkpoint}")
        model = ICIC2024LightningModel.load_from_checkpoint(
            resume_checkpoint,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            precision=args.precision,
            mixup_p=args.mixup_p,
            mixup_alpha=args.mixup_alpha,
            no_mixup_epochs=args.no_mixup_epochs,
            label_smoothing=args.label_smoothing,
            pos_weight=args.pos_weight,
            drop_path_rate=args.drop_path_rate,
        )
        # copy pretrain ema model to model
        with torch.no_grad():
            for ema_v, model_v in zip(
                model.model_ema.module.state_dict().values(),
                model.model.state_dict().values(),
            ):
                model_v.copy_(ema_v)
        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{fold}"
        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
            filename="best_loss",
        )
        pauc_checkpoint = callbacks.ModelCheckpoint(
            filename="best_pauc",
            monitor="val_pauc",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            mode="max",
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_pauc",
            patience=10,
            log_rank_zero_only=True,
            mode="max",
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}/fold{fold}",
                save_dir=logdir,
                project="isic-2024-challenge",
                tags=[f"fold{fold}"],
            )
        trainer = Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=2.0,
            precision=args.precision,
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                pauc_checkpoint,
                lr_monitor,
                early_stopping,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
            accumulate_grad_batches=args.accumulate_grad_batches,
            val_check_interval=0.5,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())

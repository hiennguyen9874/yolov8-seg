import os
import sys
import random
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder


class WarpModel(nn.Module):
    def __init__(
        self,
        weights: str = "yolov8l-seg.pt",
        nc: int = 80,
        dynamic: bool = False,
        export: bool = True,
    ):
        super().__init__()
        self.nc = nc

        self.model = YOLO(weights)
        self.model = deepcopy(self.model.model)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.float()
        self.model = self.model.fuse()
        for k, m in self.model.named_modules():
            if isinstance(m, (Detect, RTDETRDecoder)):
                m.dynamic = dynamic
                m.export = export
                m.format = "onnx"
            elif isinstance(m, C2f):
                m.forward = m.forward_split

    def forward(self, x):
        preds, protos = self.model(x)
        preds = preds.permute((0, 2, 1))

        boxes = preds[:, :, :4]
        classes = preds[:, :, 4 : self.nc + 4]
        masks = preds[:, :, self.nc + 4 :]

        return (
            torch.cat(
                (
                    boxes,
                    torch.ones(
                        (boxes.shape[0], boxes.shape[1], 1),
                        device=boxes.device,
                        dtype=boxes.dtype,
                    ),
                    classes,
                    masks,
                ),
                dim=2,
            ),
            protos,
        )

    @property
    def stride(self):
        return self.model.stride

    @property
    def names(self):
        return self.model.names

import logging
import math
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import clip
import os
import torch.nn.functional as F

from modules.ppn.archs.fcn_arch import FCNHead
from modules.ppn.archs.unet_arch import AttrUNet

logger = logging.getLogger('base')


class PPN():
    """Target Parsing Predict Network.
    """

    def __init__(self, num_classes=18):
        self.device = torch.device('cuda')
        self.num_classes = num_classes

        clip_model, _ = clip.load('ViT-L/14', device=torch.device("cpu"))
        self.clip = clip_model.to(self.device)
        self.encoder = AttrUNet(in_channels=self.num_classes, attr_embedding=768).to(self.device)
        self.decoder = FCNHead(
            in_channels=64,
            in_index=4,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            align_corners=False,
        ).to(self.device)

        self.palette = self.get_palette(self.num_classes)

    def inference(self, parsing, instruction):
        self.encoder.eval()
        self.decoder.eval()

        src_parsing = parsing.to(torch.int64)
        src_parsing_tensor = F.one_hot(src_parsing, self.num_classes).permute(0, 3, 1, 2).to(self.device)
        text_inputs = torch.cat([clip.tokenize(instruction)]).to(self.device)

        with torch.no_grad():
            text_embedding = self.clip.encode_text(text_inputs)
            text_enc = self.encoder(src_parsing_tensor, text_embedding)
            seg_logits = self.decoder(text_enc)
        seg_pred = seg_logits.argmax(dim=1)
        palette_pred = self.palette_result(seg_pred.cpu().numpy())
        palette_origin = self.palette_result(src_parsing.cpu().numpy())

        return seg_pred, palette_origin, palette_pred

    def load_network(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)

        self.encoder.load_state_dict(
            checkpoint['encoder'], strict=True)
        self.encoder.eval()

        self.decoder.load_state_dict(
            checkpoint['decoder'], strict=True)
        self.decoder.eval()

    def palette_result(self, result):
        seg = result[0]
        palette = np.array(self.palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            # import pdb;pdb.set_trace()
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]
        return color_seg
    
    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return [palette[i:i + 3] for i in range(0, len(palette), 3)]

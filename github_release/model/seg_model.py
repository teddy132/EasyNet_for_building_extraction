from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from model.seg_feature import SegmentationModel
from decoder.decoder_test import Commondecoder,Easy_Net


class EasyNet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = ( 256,128, 64, 32,16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        encoder_type= None,

    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        if encoder_type == 'easy':

            self.decoder =Easy_Net (
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            layers=[1, 1, 1, 1, 1], block=Commondecoder
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation='sigmoid',
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()




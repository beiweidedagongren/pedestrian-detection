# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""


from .HAttention import HAT
from  ultralytics.nn.modules.DAT import DAttentionBaseline
from .CPMS import CPMS
from .LSKAttention import LSKA
from .EMCA import EMCA
from .ascpa import ASCPA
from .OSRAAttention import OSRAAttention
from .MSFE import MSFE
from .BSAM import BSAM
from .FASFFHead import *
from .ModifiedSPPF import ModifiedSPPF
from .CARAFE import CARAFE
from .C2f_DWRSeg import C2f_DWRSeg
from .MDConvFusion import MDConvFusion
from .FIVEFASFF import *

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    BiLevelRoutingAttention,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    CoordAtt,
    SPDConv,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "HAT",
    "BSAM",
    "DAttentionBaseline",
    "CPMS",
    "LSKA",
    "EMCA",
    "ASCPA",
    "OSRAAttention", #10分钟一个周期
    "MSFE",
    "SPDConv",
    "ModifiedSPPF",
    "CARAFE",
    "C2f_DWRSeg",
    "MDConvFusion"
)


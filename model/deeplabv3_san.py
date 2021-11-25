from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import mobilenetv3
from torchvision.models import resnet
import model.san as san
from model.aspp_san import DeepLabHead_SAN, DeepLabV3_SAN
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models.segmentation.lraspp import LRASPP


def _segm_model(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    if 'san' in backbone_name:
        backbone = san.__dict__[backbone_name](pretrained=pretrained_backbone)
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    elif 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, _dilated=True).features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}
    if aux:
        return_layers[aux_layer] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    model_map = {
        'deeplabv3_san': (DeepLabHead_SAN, DeepLabV3_SAN),
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, num_classes, aux_loss, **kwargs):
    kwargs["pretrained_backbone"] = pretrained
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    return model


def deeplabv3_san19(pretrained=False, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a SAN-19 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Pascal VOC 2012 (21 classes)
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'san19', pretrained, num_classes, aux_loss, **kwargs)


def deeplabv3_san12(pretrained=False, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a SAN-19 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Pascal VOC 2012 (21 classes)
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'san12', pretrained, num_classes, aux_loss, **kwargs)


def deeplabv3_res50_aspp_san(pretrained=True, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a SAN ASPP head.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Pascal VOC 2012 (21 classes)
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3_san', 'resnet50', pretrained, num_classes, aux_loss, **kwargs)


def deeplabv3_res101_aspp_san(pretrained=True, num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a SAN ASPP head.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Pascal VOC 2012 (21 classes)
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3_san', 'resnet101', pretrained, num_classes, aux_loss, **kwargs)
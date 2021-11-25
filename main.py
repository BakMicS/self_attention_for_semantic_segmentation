from pathlib import Path

import os
import torch
from datetime import datetime
from sklearn.metrics import f1_score
from torch.utils import data
from torchvision import transforms
from VOC_dataset import VOCSegmentationAlt

from model.deeplabv3_san import deeplabv3_san19, deeplabv3_san12, deeplabv3_res50_aspp_san, deeplabv3_res101_aspp_san
from model2.deeplabv3_locrel import deeplabv3_lrnet101, deeplabv3_lrnet50
from torchvision.models.segmentation.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from train_model import train_model
from plot_traning_log import plot_and_save_learning_curves_from_log_file

OUT_PATH = './train_output/'


def main(args):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    if args.checkpoint:
        model = torch.load(args.checkpoint)
    elif args.model == 'deeplab50':
        pretrained = False
        model = deeplabv3_resnet50(pretrained=pretrained)
    elif args.model == 'deeplab101':
        pretrained = True
        model_trained = deeplabv3_resnet101(pretrained=pretrained)
        model = deeplabv3_resnet101(pretrained=False)
        model.backbone = model_trained.backbone
    elif args.model == 'san19':
        pretrained = False
        model = deeplabv3_san19(pretrained=pretrained)
    elif args.model == 'san12':
        pretrained = False
        model = deeplabv3_san12(pretrained=pretrained)
    elif args.model == 'deeplab50_aspp_san':
        pretrained = True
        model = deeplabv3_res50_aspp_san(pretrained=pretrained)
    elif args.model == 'deeplab101_aspp_san':
        pretrained = True
        model = deeplabv3_res101_aspp_san(pretrained=pretrained)
    elif args.model == 'deeplabv3_lrnet101':
        pretrained = False
        model = deeplabv3_lrnet101(pretrained=pretrained)
    elif args.model == 'deeplabv3_lrnet50':
        pretrained = False
        model = deeplabv3_lrnet50(pretrained=pretrained)
    else:
        raise Exception("Given model unrecognized")

    model.train()
    # Create the experiment directory if not present
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S_")
    out_dir = Path(os.path.join(OUT_PATH, date_time + args.out_dir))
    if not out_dir.exists():
        out_dir.mkdir()

    # Specify the loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score}

    # Create the dataloader
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = VOCSegmentationAlt(root='./VOC2012', image_set='train',
                                   transform=train_transform, target_transform=train_transform)
    val_transform = transforms.Compose([transforms.ToTensor()])
    val_set = VOCSegmentationAlt(root='./VOC2012', image_set='val',
                                 transform=val_transform, target_transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    dataloaders = {'Train': train_loader, 'Test': val_loader}

    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=out_dir,
                    metrics=metrics,
                    num_epochs=args.epochs,
                    pretrained=pretrained)

    # Save the trained model
    torch.save(model, out_dir / 'final_best_weights.pt')
    plot_and_save_learning_curves_from_log_file(out_dir, args)


if __name__ == "__main__":

    import argparse
    default_batch_size = 6

    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('-config', type=str, default='config/imagenet/imagenet_san10_pairwise.yaml', help='config file')
    parser.add_argument('-batch_size', type=int, default=default_batch_size)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-out_dir', type=str, default='deeplab101_asppsan_full_4_8_12_k7_512x512_batch{}_backbone_freeze'.format(default_batch_size))
    parser.add_argument('-model', type=str, default='deeplab101_aspp_san',
                        choices=['san12', 'san19', 'deeplab50', 'deeplab101', 'deeplab50_aspp_san', 'deeplab101_aspp_san',
                                 'deeplabv3_lrnet101', 'deeplabv3_lrnet50'])
    parser.add_argument('-checkpoint', type=str, default=None)
    args = parser.parse_args()

    main(args)

import time

import numpy as np
import torch
from tqdm import tqdm
from train_model import compute_miou
from sklearn.metrics import confusion_matrix


def test_model(model, test_dataloader, metrics):
    since = time.time()

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = [f'Test_{m}' for m in metrics.keys()]

    batchsummary = {a: [0] for a in fieldnames + ['Test_confusion_matrix', 'Train_confusion_matrix']}

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for sample in tqdm(iter(test_dataloader)):
        inputs, masks = [data.to(device) for data in sample]
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            y_pred = outputs['out'].data.cpu().numpy().argmax(1)
            y_true = masks.data.cpu().numpy()
            y_pred = y_pred[y_true != 255]
            y_true = y_true[y_true != 255]
            for name, metric in metrics.items():
                if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    batchsummary[f'Test_{name}'].append(
                        metric(y_true, y_pred, labels=np.arange(21), average='micro'))
                else:
                    batchsummary[f'Test_{name}'].append(
                        metric(y_true, y_pred, labels=np.arange(21), average='weighted'))
            batchsummary[f'Test_confusion_matrix'].append(
                confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(21)))

    batchsummary[f'Test_mIOU'] = compute_miou(np.sum(batchsummary[f'Test_confusion_matrix'][1:], axis=0))
    del batchsummary[f'Test_confusion_matrix']
    for field in fieldnames:
        if not field.__contains__('confusion_matrix'):
            batchsummary[field] = np.mean(batchsummary[field])
    print(batchsummary)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from model.deeplabv3_san import deeplabv3_san19
    from torchvision.models.segmentation.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from sklearn.metrics import f1_score, jaccard_score
    from torchvision import transforms
    from VOC_dataset import VOCSegmentationAlt

    batch_size = 8

    # model = deeplabv3_resnet101(pretrained=True)
    # model = torch.load('./train_output/15-07-2021_14-09-58_deeplab50_512x512_batch_6/weights.pt')
    # model = torch.load('./train_output/16-07-2021_00-58-23_san19_72x72_batch_4/model_epoch_10_ckpt.pt')
    # model = torch.load('./train_output/12-08-2021_08-08-56_10-08-2021_23-16-41_deeplab101_asppsan_conv1_512x512_batch8_backbone_freeze/model_miou_epoch_44_ckpt.pt')
    # model = torch.load('./train_output/10-10-2021_18-54-09_deeplab101_asppsan_full_4_8_12_512x512_batch8_backbone_freeze/model_miou_epoch_41_ckpt.pt')

    # Pairwise:
    # model = torch.load('./train_output/10-10-2021_18-54-09_deeplab101_asppsan_full_4_8_12_512x512_batch8_backbone_freeze/model_miou_epoch_41_ckpt.pt')
    # model = torch.load('./train_output/12-10-2021_08-48-06_deeplab101_asppsan_full_k5_4_8_12_512x512_batch6_backbone_freeze/model_miou_epoch_50_ckpt.pt')

    # Patchwise:
    # model = torch.load('./train_output/21-11-2021_11-14-15_deeplab101_asppsan_6_12_18_k3_512x512_batch8_backbone_freeze/model_miou_epoch_45_ckpt.pt')
    # model = torch.load('./train_output/17-11-2021_22-52-11_deeplab101_asppsan_full_patch_6_12_18_k5_512x512_batch10_backbone_freeze/model_miou_epoch_49_ckpt.pt')
    model = torch.load('./train_output/16-11-2021_22-15-23_deeplab101_asppsan_full_patch_6_12_18_k7_512x512_batch6_backbone_freeze/model_miou_epoch_45_ckpt.pt')

    # Base:
    # model = torch.load('./train_output/19-11-2021_07-46-08_deeplab101_asppsan_test_6_12_18_k5_512x512_batch8_backbone_freeze/model_miou_epoch_49_ckpt.pt')
    # model = torch.load('./train_output/20-11-2021_20-40-08_deeplab101_asppsan_test_6_12_18_k7_512x512_batch6_backbone_freeze/model_miou_epoch_10_ckpt.pt')

    # model.load_state_dict(checkpoint['state_dict'])

    print('Params: {:0.1f}M'.format(sum(p.numel() for p in model.parameters())/1e6))

    metrics = {'f1_score': f1_score}

    val_transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
    test_set = VOCSegmentationAlt(root='./VOC2012', image_set='val',
                                  transform=val_transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_model(model, test_loader, metrics)

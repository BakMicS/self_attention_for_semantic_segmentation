import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def compute_miou(confusion_matrix):
    # compute mean iou
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.nanmean(IoU)*100


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs, num_classes=21, ignore_index=255, pretrained=True):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_mIOU = 0
    last_epoch_wts_saved = 0
    last_epoch_miou_saved = 0

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    lr = scheduler.get_last_lr()

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'lr', 'Train_loss', 'Test_loss', 'Test_mIOU', 'Train_mIOU'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames + ['Test_confusion_matrix', 'Train_confusion_matrix']}

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
                if pretrained:  # If pretrained backbone - remove backbone from training
                    for part in model.backbone.children():
                        part.requires_grad_(False)
                    model.backbone.eval()
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):

                inputs, masks = [data.to(device) for data in sample]
                optimizer.zero_grad()  # zero the parameter gradients

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().argmax(1)
                    y_true = masks.data.cpu().numpy()
                    y_pred = y_pred[y_true != ignore_index]
                    y_true = y_true[y_true != ignore_index]
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true, y_pred, labels=np.arange(num_classes), average='micro'))
                        else:  # jaccard
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true, y_pred, labels=np.arange(num_classes), average='weighted'))
                    batchsummary[f'{phase}_confusion_matrix'].append(
                        confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes)))

                    # backward + optimize (only if in training phase)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            batchsummary['lr'] = lr
            epoch_loss = loss
            epoch_miou = compute_miou(np.sum(batchsummary[f'{phase}_confusion_matrix'][1:], axis=0))
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            batchsummary[f'{phase}_mIOU'] = epoch_miou
            del batchsummary[f'{phase}_confusion_matrix']
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            if not field.__contains__('confusion_matrix'):
                batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        # log training progress
        if epoch == 1:
            with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

        # deep copy the model of loss minimal
        if phase == 'Test' and loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if epoch >= 10 or num_epochs < 10:
                torch.save(model, os.path.join(bpath, 'model_epoch_{}_ckpt.pt'.format(epoch)))
                print('best loss model saved...')
                if last_epoch_wts_saved > 0:
                    os.remove(os.path.join(bpath, 'model_epoch_{}_ckpt.pt'.format(last_epoch_wts_saved)))
                last_epoch_wts_saved = epoch

        # deep copy the model of miou maximal
        if phase == 'Test' and epoch_miou > best_mIOU and (epoch >= 10 or num_epochs < 10):
            best_mIOU = epoch_miou
            if epoch >= 10 or num_epochs < 10:
                torch.save(model, os.path.join(bpath, 'model_miou_epoch_{}_ckpt.pt'.format(epoch)))
                print('best mIOU model saved...')
                if last_epoch_miou_saved > 0:
                    os.remove(os.path.join(bpath, 'model_miou_epoch_{}_ckpt.pt'.format(last_epoch_miou_saved)))
                last_epoch_miou_saved = epoch

        scheduler.step()
        lr = scheduler.get_last_lr()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    torch.save(model, os.path.join(bpath, 'model_last_epoch_{}_ckpt.pt'.format(epoch)))
    model.load_state_dict(best_model_wts)  # load best model weights before return

    return model

import os

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models.segmentation.segmentation import deeplabv3_resnet50
from model.deeplabv3_san import deeplabv3_san19, deeplabv3_res50_aspp_san

# model.load_state_dict(checkpoint.state_dict())
# model = deeplabv3_resnet50(pretrained=True, progress=True)

model_pair = torch.load('./train_output/10-10-2021_18-54-09_deeplab101_asppsan_full_4_8_12_512x512_batch8_backbone_freeze/model_miou_epoch_41_ckpt.pt')
model_patch = torch.load('./train_output/16-11-2021_22-15-23_deeplab101_asppsan_full_patch_6_12_18_k7_512x512_batch6_backbone_freeze/model_miou_epoch_45_ckpt.pt')
base_model = torch.load('./train_output/19-11-2021_07-46-08_deeplab101_asppsan_test_6_12_18_k5_512x512_batch8_backbone_freeze/model_miou_epoch_49_ckpt.pt')
model_pair.eval()
model_patch.eval()
base_model.eval()

images_path = './VOC2012/JPEGImages/'
segmentation_path = './VOC2012/SegmentationClass/'
with open('./VOC2012/ImageSets/Segmentation/val.txt', "r") as f:
    image_names = [x.strip() for x in f.readlines()]

for image_name in image_names[-2:-1]:
    torch.cuda.empty_cache()
    input_image = Image.open(os.path.join(images_path, image_name + '.jpg'))
    segmentation_image = Image.open(os.path.join(segmentation_path, image_name + '.png'))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.float().to('cuda')
        model_pair.float().to('cuda')
        model_patch.float().to('cuda')
        base_model.float().to('cuda')

    torch.cuda.empty_cache()
    with torch.no_grad():
        output_pair = model_pair(input_batch)['out'][0]
        output_patch = model_patch(input_batch)['out'][0]
        base_output = base_model(input_batch)['out'][0]
    output_pair_predictions = output_pair.argmax(0)
    output_patch_predictions = output_patch.argmax(0)
    base_output_predictions = base_output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 20 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    segmentation_image.putpalette(colors)
    r_pair = Image.fromarray(output_pair_predictions.byte().cpu().numpy()).resize(input_image.size)
    r_pair.putpalette(colors)
    r_patch = Image.fromarray(output_patch_predictions.byte().cpu().numpy()).resize(input_image.size)
    r_patch.putpalette(colors)
    r_base = Image.fromarray(base_output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r_base.putpalette(colors)

    plt.figure(image_name, figsize=[25,5])
    plt.subplot(151)
    plt.imshow(input_image)
    plt.subplot(152)
    plt.imshow(segmentation_image)
    plt.axis('off')
    plt.subplot(153)
    plt.imshow(r_base)
    plt.axis('off')
    plt.subplot(154)
    plt.imshow(r_pair)
    plt.axis('off')
    plt.subplot(155)
    plt.imshow(r_patch)
    plt.axis('off')
    plt.show()

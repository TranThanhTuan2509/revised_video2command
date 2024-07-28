import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchinfo import summary
import torch
import cv2
import numpy as np

image_path = "/home/tuan/Pictures/academy_argyle_v_cheltenham_0415.jpg"
image = cv2.imread(image_path)
image_size = 224
# Preprocess image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (image_size, image_size))
image = np.transpose(image, (2, 0, 1))/255.
# image = np.expand_dims(image, axis=0)
image = image[None, :, :, :]
image = torch.from_numpy(image).float()

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model.fc = nn.Linear(model.fc.in_features, 2048)
model.fc = None
# print(nn.Sequential(*list(model.children())))
# summary(model, input_size=(1, 3, 224, 224))


model.eval()
with torch.no_grad():
    output = model(image)
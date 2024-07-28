import torch.nn as nn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchinfo import summary
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
image_path = "/home/tuan/Documents/Code/VIDRoBo/Key_frames/keyFrames/keyframe44.jpg"
image = cv2.imread(image_path)
image_size = 224
original_image = image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (image_size, image_size))
image = np.transpose(image, (2, 0, 1)) / 255.0
image = torch.from_numpy(image).unsqueeze(0).float()
# Load the model
mask_rcnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
# print(nn.Sequential(*list(mask_rcnn.children())))
# summary(mask_rcnn, input_size=(1, 3, 224, 224))

# Modify the mask head
# mask_rcnn.roi_heads.mask_predictor = nn.Sequential(
#     nn.Conv2d(256, 2048, kernel_size=1),
#     nn.Sigmoid(),
#     nn.AdaptiveAvgPool2d((1, 1)),
#     nn.Flatten(),
#     nn.Linear(2048, 2048)
# )

mask_rcnn.eval()

# Perform inference
with torch.no_grad():
    output = mask_rcnn(image)
# print(output[0]['masks'])

# Reshape the masks
# if 'masks' in output[0] and output[0]['masks'].size(0) > 0:
#     masks = output[0]['masks']
#     masks_flattened = masks.view(1, -1)
#     print("Flattened masks shape:", masks_flattened.shape)
# else:
#     print("No masks detected.")
# masks = output[0]['masks'] > 0.5 # Threshold to get binary masks
# if masks.size(0) > 0:
#     mask = masks[0, 0].mul(255).byte().cpu().numpy()
#     cv2.imshow('Mask', mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

masks = output[0]['masks'] > 0.5
scores = output[0]['scores']
boxes = output[0]['boxes']
copy = np.zeros_like(original_image)
for i in [1, 2]:
    mask = masks[i, 0].mul(255).byte().cpu().numpy()
    score = scores[i].item()
    box = boxes[i].cpu().numpy().astype(int)

    # Create a copy of the original image to overlay the mask
    mask_image = original_image.copy()

    # Resize mask to match the original image size
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    # Create a colored overlay for the mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask_resized > 127] = [255, 255, 255]  # Green color for mask
    copy += colored_mask
    # # Combine original image with the mask overlay
    # combined_image = cv2.addWeighted(mask_image, 1, colored_mask, 0.5, 0)
    #
    # # Draw the bounding box
    # cv2.rectangle(combined_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #
    # # Add the score as text
    # cv2.putText(combined_image, f'Score: {score:.2f}', (box[0], box[1] - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Plot the image with the mask
combined_image = np.multiply(copy, mask_image)
plt.figure(figsize=(8, 8))
plt.imshow(combined_image)
plt.title(f'Object {i + 1}')
plt.axis('off')
plt.show()
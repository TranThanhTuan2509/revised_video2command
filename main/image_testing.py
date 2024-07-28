# import numpy as np
# import cv2
#
#
# def display_npy_image(data):
#     image_data = np.load(data)
#     print(image_data.shape)
#
#
# # Example usage
# file_path = '/home/tuan/Documents/Code/video2command/resnet50_keras_feature_no_sub_mean1/P03_cam01_P03_coffee_14.npy'
# display_npy_image(file_path)
# import pickle
#
# with open('/home/tuan/Documents/Code/video2command/train_test_split/train_resnet50_keras_feature_no_sub_mean.pkl', 'rb') as f:
#     x = pickle.load(f)
# print(x.loc[5])
# print(x.loc[5]["video_path"])

import numpy as np

x = np.array([])
print(x.ndim)
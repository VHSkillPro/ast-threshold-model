import cv2, os
import numpy as np
from sklearn.model_selection import train_test_split
from ThresholdModel import ThresholdModel

# # images_path = os.path.join(os.getcwd(), 'dataset/ImageSets/')
# # annotations_path = os.path.join(os.getcwd(), 'dataset/SegmentationClass/')

# # images_name = os.listdir(images_path)
# # annotations_name = os.listdir(annotations_path)

# # images_name_train, images_name_test, annotations_name_train, annotations_name_test = train_test_split(images_name, annotations_name, train_size=0.8)

# # print("Split dataset OK")

# # images_train = [] 
# # annotations_train = []

# # images_test = []
# # annotations_test = []

# # for image_name in images_name_train:
# #     images_train.append(cv2.imread(os.path.join(images_path, image_name)))

# # for image_name in images_name_test:
# #     images_test.append(cv2.imread(os.path.join(images_path, image_name)))
    
# # for annotation_name in annotations_name_train:
# #     annotations_train.append(cv2.imread(os.path.join(annotations_path, annotation_name)))

# # for annotation_name in annotations_name_test:
# #     annotations_test.append(cv2.imread(os.path.join(annotations_path, annotation_name)))
    
# # print("Read images OK")
    
# # model = ThresholdModel()
# # model.add_label('DiaChua', [255, 221, 51])
# # model.add_label('VungVoKhuan', [102, 255, 102])
# # model.add_label('GiayKhangSinh', [240, 120, 240])

# # model.fit(images_train, annotations_train)

# # print("Train model OK")

sum = np.array([0, 0, 0])
x = np.array([[1, 0, 1], [0, 2, 1], [3, 3, 3], [4, 4, 4]])

print(np.sum(x > 0, axis=0))

# mean = x.mean(axis=0)
# print(np.sqrt(np.sum((x - mean) ** 2, axis=0) / len(x)))

# print(x.std(axis=0))
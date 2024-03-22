import os
import cv2
import numpy as np

class ThresholdModel:  
    def __init__(self, images_path: str, annotations_path: str) -> None:
        self.label_colors = dict()
        self.threshold = dict()
        self.images_path = images_path
        self.annotations_path = annotations_path
    
    def __handle_image(self, image_name: str, annotation_name: str, label_name: str):
        image = cv2.imread(os.path.join(self.images_path, image_name))
        annotation = cv2.imread(os.path.join(self.annotations_path, annotation_name))
        
        intensity = image[np.where(np.all(annotation == self.label_colors[label_name], axis=2))]
        return (intensity[:, 0], intensity[:, 1], intensity[:, 2])
        
    def add_label(self, label_name: str, label_color: list) -> None:
        self.label_colors[label_name] = label_color
        self.threshold[label_name] = {
            'min': [0, 0, 0],
            'max': [0, 0, 0]
        }
    
    def fit(self, images_name: list[str], annotations_name: list[str]) -> None:
        for label in self.label_colors.keys():
            B_sum, G_sum, R_sum = 0, 0, 0
            B_cnt, G_cnt, R_cnt = 0, 0, 0
            
            for image, annotation in zip(images_name, annotations_name):
                res = self.__handle_image(image, annotation, label)
                
                B_sum += np.sum(res[0]); B_cnt += np.size(res[0])
                G_sum += np.sum(res[1]); G_cnt += np.size(res[1])
                R_sum += np.sum(res[2]); R_cnt += np.size(res[2])
                
            B_mean = B_sum / B_cnt
            G_mean = G_sum / G_cnt
            R_mean = R_sum / R_cnt
            
            B_sum_square, G_sum_square, R_sum_square = 0, 0, 0
            
            for image, annotation in zip(images_name, annotations_name):
                res = self.__handle_image(image, annotation, label)
                
                B_sum_square += np.sum((res[0] - B_mean) ** 2)
                G_sum_square += np.sum((res[1] - G_mean) ** 2)
                R_sum_square += np.sum((res[2] - R_mean) ** 2)

            B_stddev = np.sqrt(B_sum_square / B_cnt)
            G_stddev = np.sqrt(G_sum_square / G_cnt)
            R_stddev = np.sqrt(R_sum_square / R_cnt)
            
            self.threshold[label]['min'] = [B_mean - B_stddev, G_mean - G_stddev, R_mean - R_stddev]
            self.threshold[label]['max'] = [B_mean + B_stddev, G_mean + G_stddev, R_mean + R_stddev]
            
            print(label, self.threshold[label]['min'], self.threshold[label]['max'])

    def predict(self, image: np.ndarray, label: str) -> np.ndarray:
        mask = image.copy()
        pos = (self.threshold[label]['min'] <= mask).all(axis=2) & (mask <= self.threshold[label]['max']).all(axis=2)
        mask[np.where(pos)] = self.label_colors[label]
        mask[np.where(np.logical_not(pos))] = [0, 0, 0]
        return mask
import os, cv2
from re import S
import numpy as np

class ThresholdModel:  
    def __init__(self, images_path: str, annotations_path: str) -> None:
        self.label_colors = dict()
        self.threshold = dict()
        self.__statistic = dict()
        self.images_path = images_path
        self.annotations_path = annotations_path
    
    def add_label(self, label_name: str, label_color: list[3]) -> None:
        self.label_colors[label_name] = label_color
        
        self.threshold[label_name] = {
            'min': np.array([0, 0, 0], dtype=np.float64),
            'max': np.array([0, 0, 0], dtype=np.float64)
        }
        
        self.__statistic[label_name] = {
            'size': np.array([0, 0, 0], dtype=np.float64),
            'sum': np.array([0, 0, 0], dtype=np.float64),
            'dev_square': np.array([0, 0, 0], dtype=np.float64),
            'mean': np.array([0, 0, 0], dtype=np.float64),
            'std': np.array([0, 0, 0], dtype=np.float64)
        }
        
    def fit(self, images_name: list[str], annotations_name: list[str]) -> None:
        for image_name, annotation_name in zip(images_name, annotations_name):
            image = cv2.imread(os.path.join(self.images_path, image_name))
            annotation = cv2.imread(os.path.join(self.annotations_path, annotation_name))
            
            for label_name in self.label_colors.keys():
                pixels = image[np.where(np.all(annotation == self.label_colors[label_name], axis=2))]
                self.__statistic[label_name]['sum'] += pixels.sum(axis=0)
                self.__statistic[label_name]['size'] += np.sum(pixels > 0, axis=0)
            
        for key, item in self.__statistic.items():
            item['mean'] = item['sum'] / item['size']
                
        for image_name, annotation_name in zip(images_name, annotations_name):
            image = cv2.imread(os.path.join(self.images_path, image_name))
            annotation = cv2.imread(os.path.join(self.annotations_path, annotation_name))
            
            for label_name in self.label_colors.keys():
                item = self.__statistic[label_name]
                pixels = image[np.where(np.all(annotation == self.label_colors[label_name], axis=2))]
                item['dev_square'] += np.sum((pixels - item['mean']) ** 2, axis=0)
        
        for label_name in self.label_colors.keys():
            item = self.__statistic[label_name]
            item['std'] = np.sqrt(item['dev_square'] / item['size'])
            
            self.threshold[label_name]['min'] = item['mean'] - item['std']
            self.threshold[label_name]['max'] = item['mean'] + item['std']
            
            print(label_name, self.threshold[label_name]['min'], self.threshold[label_name]['max'])

    def predict(self, image: np.ndarray, label: str) -> np.ndarray:
        mask = image.copy()
        pos = (self.threshold[label]['min'] <= mask).all(axis=2) & (mask <= self.threshold[label]['max']).all(axis=2)
        mask[np.where(pos)] = self.label_colors[label]
        mask[np.where(np.logical_not(pos))] = [0, 0, 0]
        return mask
    
    def get_mask(self, mask: np.ndarray, label_name: str) -> np.ndarray:
        mask_label = mask.copy()
        mask_label[np.where(np.all(mask_label != self.label_colors[label_name], axis=2))] = [0, 0, 0]
        return mask_label
    
# def pixel_accuracy(predict: np.ndarray, mask: np.ndarray):
    # return np.sum(predict == mask) / np.size(mask)
    
def iou_measure(predict: np.ndarray, mask: np.ndarray):
    intersect = np.sum(np.all((predict & mask) != [0, 0, 0], axis=2))
    union = np.sum(np.all((predict | mask) != [0, 0, 0], axis=2))
    return intersect / union
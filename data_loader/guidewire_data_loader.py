import os
import cv2
import numpy
import random
import numpy as np
from torch.utils import data
from augmentation import pixel_coords as aug_functions

'''
todo

1. normalize the image and label:
        - For image, normalize with mean and std
        - For label, normalize to [0, 1]

2. add random masking to the image.
'''

class GuidewireDataSet(data.Dataset):
    def __init__(self, data_sample_names: list, apply_augmentation: bool = False, config: dict = None):
        self.data_sample_names = data_sample_names
        self.apply_augmentation = apply_augmentation
        self.config = config
        self.augmentation_config = self.config['dataset']['augmentation']
        self.image_initial_resize_ratio = self.config['dataset']['image_initial_resize_ratio']

    def __len__(self):
        return len(self.data_sample_names)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_sample_names[index]
        image_cv = cv2.imread(image_path).astype(np.float32)
        print(f"image_cv data type: {type(image_cv)}")
        # from BGR to RGB, and normalize to [0, 1]
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) / 255.0
        height, width = image.shape[:2]
        # read the label from txt file.
        # inside the txt file, the label is in the format of [1, x, y]
        label = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            info = line.split(' ')
            # the label starts with dummy value 1, so we ignore
            # info[0] and start from info[1]
            # the raw coordinates are normalized 
            # by the width and height of the image
            label.append(int(info[1]) / (width-1))
            label.append(int(info[2]) / (height-1))
        label = numpy.array(label)
        # resize the image to the initial size
        image = cv2.resize(image, (round(width * self.image_initial_resize_ratio), round(height * self.image_initial_resize_ratio)))
        if self.apply_augmentation:
            image, label = self.augment_sample(image, label)
        return image, label
    
    def augment_sample(self, image, label):
        """
        Apply augmentations to image and label based on config
        Returns augmented image and label
        """
        image_augmented = image.copy()
        label_augmented = label.copy()
        # Apply each augmentation with its probability
        for aug_name, aug_params in self.augmentation_config.items():
            if random.random() < aug_params['probability']:
                if aug_name == 'random_brightness':
                    image_augmented, label_augmented = aug_functions.augment_random_brightness(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_horizontal_flip':
                    image_augmented, label_augmented = aug_functions.augment_horizontal_flip(
                        image_augmented, label_augmented,
                    ) # no args
                elif aug_name == 'random_vertical_flip':
                    image_augmented, label_augmented = aug_functions.augment_vertical_flip(
                        image_augmented, label_augmented,
                    ) # no args
                elif aug_name == 'random_gaussian_noise':
                    image_augmented, label_augmented = aug_functions.augment_random_gaussian_noise(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_resize':
                    image_augmented, label_augmented = aug_functions.augment_random_resize(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_shear':
                    image_augmented, label_augmented = aug_functions.augment_random_shear(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_rotation':
                    image_augmented, label_augmented = aug_functions.augment_random_rotation(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_scale_intensity':
                    image_augmented, label_augmented = aug_functions.augment_random_scale_intensity(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_saturation':
                    image_augmented, label_augmented = aug_functions.augment_random_saturation(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_hue_shift':
                    image_augmented, label_augmented = aug_functions.augment_random_hue_shift(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_gaussian_sharpness':
                    image_augmented, label_augmented = aug_functions.augment_random_gaussian_sharpness(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_elastic_deformation':
                    image_augmented, label_augmented = aug_functions.augment_random_elastic_deformation(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_cutout':
                    image_augmented, label_augmented = aug_functions.augment_random_cutout(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_perspective':
                    image_augmented, label_augmented = aug_functions.augment_random_perspective(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                elif aug_name == 'random_crop':
                    image_augmented, label_augmented = aug_functions.augment_random_crop(
                        image_augmented, label_augmented,
                        **aug_params['args']
                    )
                else:
                    raise ValueError(f"Unknown augmentation name: {aug_name}")

        return image_augmented, label_augmented


class GuidewireDataLoader():
    def __init__(self, dir_dataset: str, split_ratio: list):
        self.dir_dataset = dir_dataset
        assert len(split_ratio) == 3
        self.data_sample_names = self._load_data(self.dir_dataset)
        # shuffle data_sample_names tuple for 1000 times
        # for reproducibility, set random seed here
        random.seed(42)
        for _ in range(100): # random shuffle
            random.shuffle(self.data_sample_names)
        n_train = round(len(self.data_sample_names) * split_ratio[0])
        n_val = round(len(self.data_sample_names) * split_ratio[1])
        n_test = len(self.data_sample_names) - n_train - n_val
        self.train_data_sample_names = self.data_sample_names[:n_train]
        self.val_data_sample_names = self.data_sample_names[n_train:n_train + n_val]
        self.test_data_sample_names = self.data_sample_names[n_train + n_val:]
        
    def _load_data(self, dir_dataset: str):
        data_sample_names = []
        # Enumerate all images and labels
        for video_dir in os.listdir(dir_dataset):
            video_path = os.path.join(dir_dataset, video_dir)
            labels_dir = os.path.join(video_path, 'Labels')
            images_dir = os.path.join(video_path, 'Images')
            for label_name in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_name)
                image_path = os.path.join(images_dir, label_name.replace('.txt', '.jpg'))
                data_sample_names.append((image_path, label_path))
        return data_sample_names
    
    def get_data_sample_names(self, data_type: str):
        assert data_type in ['train', 'val', 'test']
        if data_type == 'train':
            return self.train_data_sample_names
        elif data_type == 'val':
            return self.val_data_sample_names
        elif data_type == 'test':
            return self.test_data_sample_names
        else:
            raise ValueError(f"Invalid data type: {data_type}")
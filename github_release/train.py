import os
from RSRnetwork import RSRNet
# from network import UNetBranch_Network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start = time.time()
print(start)
DATA_DIR = r'D:\hhg\paper_experiment\data\WHU'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('no data!')

x_train_dir = os.path.join(DATA_DIR, 'train_img')
y_train_dir = os.path.join(DATA_DIR, 'train_label')

x_valid_dir = os.path.join(DATA_DIR, 'test_img')
y_valid_dir = os.path.join(DATA_DIR, 'test_label')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """


    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        # self.ids.sort(key=lambda x:int(x[:-4]))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = 255

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


# Lets look at data we have

import albumentations as albu
import random

random.seed(100)


def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.6),

        albu.RandomRotate90(p=0.6),

        albu.ShiftScaleRotate(scale_limit=0.5, shift_limit=0.5, rotate_limit=180, p=0.6),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Flip(p=0.6),

        albu.RandomRotate90(p=0.6),

        albu.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.2, rotate_limit=0, p=0.6),

    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]

    return albu.Compose(_transform)


#### Visualize resulted augmented images and masks


import torch
import numpy as np
import segmentation_models_pytorch as smp1
from model import seg_model


ENCODER1 ='timm-regnety_008'  # timm-regnety_008,mobileone_s1,mobilenet_v2
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['build']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

#Easy-Net
model = seg_model.EasyNet(
    encoder_name=ENCODER1,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
    encoder_depth=4,
    decoder_channels=[128,64,32],encoder_type='easy'
)

model = model.cuda()
preprocessing_fn = smp1.encoders.get_preprocessing_fn(ENCODER1, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=None,
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
loss1 = smp1.utils.losses.DiceLoss()

metrics = [
    smp1.utils.metrics.IoU(threshold=0.5),
    smp1.utils.metrics.Accuracy(threshold=0.5),
    smp1.utils.metrics.Recall(threshold=0.5),
    smp1.utils.metrics.Precision(threshold=0.5),
    smp1.utils.metrics.Fscore(threshold=0.5)

]

lr_list = []
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=19, verbose=True,
                                                       cooldown=0, min_lr=0.00001)

train_epoch = smp1.utils.train.TrainEpoch(

    model,
    loss=loss1,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp1.utils.train.ValidEpoch(
    model,
    loss=loss1,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)



max_score = 0
train_loss = []
train_iou = []
train_acc = []
val_loss = []
val_iou = []
val_acc = []
for i in range(0, 150):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    train_loss.append(train_logs['dice_loss'])
    train_acc.append(train_logs['accuracy'])
    train_iou.append(train_logs['iou_score'])
    if i > 50:
        valid_logs = valid_epoch.run(valid_loader)
        scheduler.step(valid_logs['iou_score'])
        val_loss.append(valid_logs['dice_loss'])
        val_acc.append(valid_logs['accuracy'])
        val_iou.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model.state_dict(), r'D:\hhg\paper_experiment\network\model\EasyNet_whu.pth')
            print('Model saved!')
end = time.time()
elapsed = (end - start) / 3600
print("Time used:", elapsed)

print(train_loss, train_iou, train_acc, val_loss, val_iou, val_acc)

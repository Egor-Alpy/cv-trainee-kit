import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(mode: str, augmentation: str = "medium"):
    """
    mode: "train" | "val"
    augmentation: "light" | "medium" | "heavy"
    """
    if mode == "val":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    if augmentation == "light":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    if augmentation == "medium":
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

    # heavy
    return A.Compose([
        A.Resize(300, 300),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, p=0.5),
        A.HueSaturationValue(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

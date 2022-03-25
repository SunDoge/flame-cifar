import flowvision.transforms as T


def CifarTransform(train: bool):
    normalize = T.Normalize(
        mean=[0.491, 0.482, 0.447],
        std=[0.247, 0.243, 0.262]
    )

    if train:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])

    return transform

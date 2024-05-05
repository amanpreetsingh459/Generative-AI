from torchvision import transforms
from torchvision.datasets import StanfordCars
from torch.utils.data import ConcatDataset


def get_stanford_cars_dataset(path, img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
    data_transform = transforms.Compose(data_transforms)

    train = StanfordCars(root=path, download=True, transform=data_transform)

    test = StanfordCars(
        root=path, download=True, transform=data_transform, split="test"
    )
    return ConcatDataset([train, test])

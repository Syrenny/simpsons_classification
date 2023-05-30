import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# Define the transformation for the images
transform = {
    "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
    "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
}


def load_dataset(config):
    dataset = ImageFolder(config["dataset_path"])
    config["num_classes"] = len(dataset.classes)
    config["name_classes"] = dataset.classes
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size])
    train_set = MyDataset(train_set, transform=transform["train"])
    test_set = MyDataset(test_set, transform=transform["test"])
    loader = {
        "train": DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True),
        "test": DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    }
    return loader, train_set, test_set

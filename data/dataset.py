import torch
from torchvision import transforms, datasets
from utils.tokenizer_utils import tokenize_texts

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

class CIFAR10WithText(torch.utils.data.Dataset):
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        text = self.class_names[label]
        text_token = tokenize_texts([text])[0]  # 返回tensor的第0个元素
        return img, text_token

def load_train_dataset():
    cifar10_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    return CIFAR10WithText(cifar10_train, cifar10_train.classes)

def load_test_dataset():
    cifar10_test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    return cifar10_test, cifar10_test.classes

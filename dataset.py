from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from PIL import Image

class MyImageFolder(ImageFolder):
    def __init__(self,data_path,transform):
        super(MyImageFolder, self).__init__(root=data_path,transform=transform)
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,index,target

class MyCIFAR10(datasets.CIFAR10):
    def __init__(self,data_path,transform,train):
        super(MyCIFAR10, self).__init__(root=data_path,download=True,transform=transform,train=train)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MyCIFAR10_PCL(datasets.CIFAR10):
    def __init__(self,data_path,transform,train):
        super(MyCIFAR10_PCL, self).__init__(root=data_path,download=True,transform=transform,train=train)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, index,target





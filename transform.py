from torchvision import transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.ToTensor()])
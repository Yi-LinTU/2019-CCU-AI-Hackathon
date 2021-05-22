from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class IMAGE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.z = []
        self.transform = transform
        self.num_classes = 0
        # print(self.root_dir.name)
        for i, _dir in enumerate(self.root_dir.glob('*')):
            # print(_dir)
            for file in _dir.glob('*'):
                self.x.append(file)
                self.y.append(i)
                self.z.append(str(file))
            self.num_classes += 1
            # print(self.num_classes)
        # print(self.num_classes)
        # print(self.x)
        # print(self.z)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # image=self.x[index]
        # print(self.x[])
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index], self.z[index]

from torch.utils.data import Dataset

class WeaponDetectionDataset(Dataset):
    def __init__(self, file_list:list, transform = None, target_transform = None):
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = self.file_list[index]["image"]
        label = self.file_list[index]["label"]
        if(self.transform):
            img = self.transform(img)
        if(self.target_transform):
            label = self.target_transform(label)
        return img, label
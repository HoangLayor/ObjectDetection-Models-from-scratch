from lib import *

class MyDataset(data.Dataset):
    def __init__(self, root, img_list, anno_list, transform=None):
        '''
        Args:
            root: str, root directory of images
            img_list: list, list of image paths
            anno_list: list, list of annotation information
            transform: torch.transforms, data augmentation
        '''

        self.transform = transform
        self.img_list = img_list
        self.anno_list = anno_list
        self.root = root

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image, annotations, height, width = self.pull_item(idx)
        return image, annotations

    def pull_item(self, idx):
        image_id, file_name, height, width = self.img_list[idx]
        annotations = self.anno_list[idx][1].copy() # deep copy

        file_path = self.root + "/" + file_name
        image = read_image(file_path, ImageReadMode.RGB) # torch.Tensor (C, H, W)
        height, width = image.shape[1:]

        for anno in annotations:
            anno[0] = anno[0] / width
            anno[1] = anno[1] / height
            anno[2] = anno[2] / width
            anno[3] = anno[3] / height 

        if self.transform is not None:
            image = self.transform(image)

        return image, annotations, height, width

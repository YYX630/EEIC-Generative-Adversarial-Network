from pathlib import Path
import pathlib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pickle
import torch
import numpy as np

class ImageDataset(Dataset):
    EXTENSIONS = [".txt"]

    def __init__(self, par_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(par_dir)
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.img_paths[index]
        try:
            img = Image.open(path.with_suffix('.png'))
        except:
            img = Image.open(path.with_suffix('.jpg'))
        img = img.convert("RGB") #モノクロの広告が存在するので次元数を合わせる
        if self.transform is not None:
            # 前処理がある場合は行う。
            img = self.transform(img)
        txt_path = path.with_suffix('.txt')
        with open(str(txt_path), "rb") as f:
            embed = pickle.load(f)
        return img, torch.tensor(embed)
    def _get_img_paths(self, par_dir):
        return list(pathlib.Path(par_dir).glob('**/*.txt'))
    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)
# # sample code
# transform=transforms.Compose([
#                           	transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
#                           	transforms.RandomHorizontalFlip(),
#                           	transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#                           	transforms.ToTensor(),
#                           	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                       	])

# dataset = ImageFolder("../all_data/data", transform)

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)

# for itr, batch in enumerate(dataloader):
#     print(itr, batch[0].size(), batch[1].size())

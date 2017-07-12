import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import sys
import pickle
import numpy as np
import nltk
from PIL import Image
from vocab import Vocabulary
sys.path.append('../../../coco/PythonAPI')
from pycocotools.coco import COCO


class CocoImages(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root,transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            transform: image transformer 
        """
        self.root = root
	self.cocoimages=os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

	path=self.cocoimages[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image

    def __len__(self):
        return len(self.cocoimages)

    
def cocoimage_data_loader(root,transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    coco = CocoImages(root=root,
                       transform=transform)
    
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              )
    return data_loader

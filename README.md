# KD_ImageCaptioning
# Follow the instructions, this repo works for Python 2.7

1. In the parent directory of this repo(../) install coco-dataset api's,
and compile the python API's by calling make to install the coco api's
local python package. https://github.com/pdollar/coco.git


2. The COCO_Dataset folder should contain the complete coco dataset as all
 scripts use this relative path. The folder should contain subfolders, namely
 train2017, val2017 and annotations.

3. Install torch vision and pytorch on Anaconda 2.7.  On cuda 8
 the command is conda install pytorch torchvision cuda80 -c soumith



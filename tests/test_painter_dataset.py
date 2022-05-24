import pytest
from PIL import Image

import torch
from torchvision import transforms

from styleclr.dataset import PainterDataset


@pytest.fixture
def painter_dataset():

    transform = transforms.ToTensor()
    return PainterDataset(root='datasets/painter', transform=transform)


def test_read_painter_csv(painter_dataset):

    painting_102257 = painter_dataset.csv['filename'] == '102257.jpg'
    assert painter_dataset.csv.loc[painting_102257, 'title'][0] == 'Uriel'  


def test_dataset_len(painter_dataset):

    assert len(painter_dataset) == 79433


def test_get_item(painter_dataset):

    image = painter_dataset[2][0]

    raw_img = Image.open('datasets/painter/train/29855.jpg')
    toTensor = transforms.ToTensor()
    tensor_img = toTensor(raw_img)

    assert torch.equal(image, tensor_img) 



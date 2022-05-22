from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import transforms, datasets

from simclr.data_aug.gaussian_blur import GaussianBlur
from simclr.data_aug.view_generator import ContrastiveLearningViewGenerator
from simclr.exceptions.exceptions import InvalidDatasetSelection

import adain.net as net

from styleclr.test import test_transform, style_transfer


class StylizedDataset:

    def __init__(self, content_dataset, style_dataset, vgg_path, decoder_path, alpha):

        self.content_dataset = content_dataset

        style_dataloader = torch.utils.data.DataLoader(style_dataset, batch_size=2, shuffle=True, num_workers=2)
        self.style_iter = iter(style_dataloader)

        self.decoder_path = Path(decoder_path)
        self.vgg_path = Path(vgg_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.content_size = 96
        self.style_size = 512
        self.crop = False
        self.alpha = alpha

        self.decoder = net.decoder
        self.vgg = net.vgg

        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.vgg.load_state_dict(torch.load(self.vgg_path))

        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        self.content_tf = test_transform(self.content_size, self.crop)
        self.style_tf = test_transform(self.style_size, self.crop)

        self.toPIL = transforms.ToPILImage()

    def __len__(self):
        return len(self.content_dataset)

    def __getitem__(self, idx):

        content_image = self.content_dataset[idx][0]
        style_images = next(self.style_iter)
        style_image1 = style_images[0][0]
        style_image2 = style_images[0][1]

        content = self.content_tf(self.toPIL(content_image))
        style1 = self.style_tf(self.toPIL(style_image1))
        style2 = self.style_tf(self.toPIL(style_image2))

        content = content.to(self.device).unsqueeze(0)
        style1 = style1.to(self.device).unsqueeze(0)
        style2 = style2.to(self.device).unsqueeze(0)

        with torch.no_grad():
            output1 = style_transfer(self.device, self.vgg, self.decoder, content, style1, self.alpha)
            output2 = style_transfer(self.device, self.vgg, self.decoder, content, style2, self.alpha)

        return (output1, output2)

        
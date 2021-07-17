import random
import numpy as np

import torchvision.transforms as T
from PIL import ImageFilter


# adapted from
# https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_transforms/img_pil_color_distortion.py
class ImgPilColorDistortion:
    def __init__(self, strength):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
        self.strength = strength
        self.color_jitter = T.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.rnd_color_jitter = T.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = T.RandomGrayscale(p=0.2)
        self.transforms = T.Compose([self.rnd_color_jitter, self.rnd_gray])

    def __call__(self, image):
        return self.transforms(image)


# adapted from
# https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_transforms/img_pil_gaussian_blur.py
class ImgPilGaussianBlur:
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p, radius_min, radius_max):
        """
        Args:
            p (float): probability of applying gaussian blur to the image
            radius_min (float): blur kernel minimum radius used by ImageFilter.GaussianBlur
            radius_max (float): blur kernel maximum radius used by ImageFilter.GaussianBlur
        """
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        should_blur = np.random.rand() <= self.prob
        if not should_blur:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class MultiViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

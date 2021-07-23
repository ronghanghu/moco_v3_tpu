import random
import numpy as np

from PIL import ImageFilter, ImageOps


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


# adapted from
# https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_transforms/img_pil_random_solarize.py
class ImgPilRandomSolarize:
    """
    Randomly apply solarization transform to an image.
    This was used in BYOL - https://arxiv.org/abs/2006.07733
    """

    def __init__(self, p):
        """
        Args:
            p (float): Probability of applying the transform
        """
        self.prob = p

    def __call__(self, img):
        should_solarize = np.random.rand() <= self.prob
        if not should_solarize:
            return img

        return ImageOps.solarize(img)


class MultiViewGenerator:
    """Take apply separate transforms to each crop."""

    def __init__(self, base_transform_list):
        self.base_transform_list = base_transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.base_transform_list]

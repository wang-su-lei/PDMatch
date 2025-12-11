# code in this file is adpated from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [
        # AutoContrast: 自动调整图像对比度，拉伸最暗/最亮像素到全范围
        # max_v=None: 不需要调整范围；bias=None: 无附加偏移
        (AutoContrast, None, None),

        # Brightness: 调整亮度
        # max_v=0.9: 最大亮度变化幅度（0.0–0.9），brightness = 1.0 ± v
        # bias=0.05: 最小偏移量，避免完全无变化
        (Brightness, 0.9, 0.05),

        # Color: 调整颜色饱和度
        # max_v=0.9: 最大饱和度变化幅度
        # bias=0.05: 最小偏移量
        (Color, 0.9, 0.05),

        # Contrast: 调整对比度
        # max_v=0.9: 最大对比度变化幅度
        # bias=0.05: 最小偏移量
        (Contrast, 0.9, 0.05),

        # Equalize: 直方图均衡化
        # max_v=None: 使用全范围均衡化；bias=None: 无附加偏移
        (Equalize, None, None),

        # Identity: 恒等映射，不做任何变化
        # max_v=None, bias=None: 占位，无实际操作
        (Identity, None, None),

        # Posterize: 降低色彩位深
        # max_v=4: 降到最多 4 位深
        # bias=4: 固定位深参数
        (Posterize, 4, 4),

        # Rotate: 随机旋转
        # max_v=30: 最大旋转角度 30°
        # bias=0: 从 -max_v 到 +max_v 随机旋转
        (Rotate, 30, 0),

        # Sharpness: 调整锐度
        # max_v=0.9: 最大锐度变化幅度
        # bias=0.05: 最小偏移量
        (Sharpness, 0.9, 0.05),

        # ShearX: X 方向剪切
        # max_v=0.3: 最大剪切程度（比例）
        # bias=0: 从 -max_v 到 +max_v 剪切
        (ShearX, 0.3, 0),

        # ShearY: Y 方向剪切
        # max_v=0.3: 最大剪切程度（比例）
        # bias=0: 从 -max_v 到 +max_v 剪切
        (ShearY, 0.3, 0),

        # Solarize: 反色处理，像素值超过阈值时取反
        # max_v=256: 阈值范围（0–256），表示使用全范围阈值
        # bias=0: 无附加偏移
        (Solarize, 256, 0),

        # TranslateX: X 方向平移
        # max_v=0.3: 最大平移幅度为图像宽度的 30%
        # bias=0: 从 -max_v 到 +max_v 随机平移
        (TranslateX, 0.3, 0),

        # TranslateY: Y 方向平移
        # max_v=0.3: 最大平移幅度为图像高度的 30%
        # bias=0: 从 -max_v 到 +max_v 随机平移
        (TranslateY, 0.3, 0)
    ]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs



class RandAugmentMC(object):
    def __init__(self, n, m, image_size=None, cutout_ratio=0.5):
        """
        n: 每张图随机变换的个数
        m: 变换强度上限（1～m）
        image_size: 图像短边或固定尺寸，用来算 cutout 长度
        cutout_ratio: cutout 长度占 image_size 的比例
        """
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.image_size = image_size
        self.cutout_ratio = cutout_ratio
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        # 随机选 n 个操作并以 50% 概率执行
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)

        # 计算 cutout 大小：要么用传入的 image_size，要么动态取图像尺寸
        if self.image_size is not None:
            length = int(self.image_size * self.cutout_ratio)
        else:
            w, h = img.size
            length = int(min(w, h) * self.cutout_ratio)

        img = CutoutAbs(img, length)
        return img
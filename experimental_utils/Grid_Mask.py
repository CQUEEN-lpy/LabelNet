import numpy as np
import math
from PIL import Image
import tensorflow as tf


class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=1, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.shape[0]
        w = img.shape[1]

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))
        # print(img.shape)

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = tf.constant(mask, dtype='float32')
        if self.mode == 1:
            mask = 1 - mask
        mask = tf.expand_dims(mask, 2)
        mask = tf.tile(mask, [1, 1, 3])
        img = img * mask

        return img


class GridMask(tf.keras.layers.Layer):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=1, prob=1., ):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def build(self, inputs):
        self.d1 = 1

    @tf.function
    def call(self, inputs, **kwargs):
        n = inputs.shape[0]
        if n is None:
            return inputs
        y = []
        for i in range(n):
            y.append(self.grid(inputs[i]))
        y = tf.convert_to_tensor(y)
        return y


from PIL import Image
import numpy as np


def blend(ir, ip):
    ip = np.asarray(ip)
    ip = np.asarray([ip, ip*255, ip]).transpose([1, 2, 0])
    ip = Image.fromarray(ip)

    ir = Image.blend(ir.convert('RGB'), ip, 0.5)

    return ir


def coloring(pred, nc):
    colors = [[0, 0, 0],
              [255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 255, 0],
              [0, 255, 255],
              [255, 0, 255],
              [255, 239, 213],
              [0, 0, 205],
              [205, 133, 63],
              [210, 180, 140]]

    pred = np.expand_dims(pred, axis=-1)
    pred = np.repeat(pred, 3, axis=-1)
    for c in range(nc):
        pred[..., 0][pred[..., 0] == c] = colors[c][0]
        pred[..., 1][pred[..., 1] == c] = colors[c][1]
        pred[..., 2][pred[..., 2] == c] = colors[c][2]

    return pred

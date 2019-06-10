
import random
import numpy as np
import cv2
def BrightnessTransform(data, delta_max = 50):
    """
    Transform brightness
    Parameters: delta
    """

    data = data.astype(np.float32)
    delta = random.randint(-delta_max, delta_max)
    data += delta
    data[data>255] = 255
    data[data<0] = 0
    data = data.astype(np.uint8)
    return data

def ContrastTransform(data, lower=0.5, upper=1.5):
    """
    Transform contrast
    Parameters: lower, upper
    """
    data = data.astype(np.float32)
    delta = random.uniform(lower, upper)
    data *= delta
    data[data>255] = 255
    data[data<0] = 0
    data = data.astype(np.uint8)
    return data


def SaturationTransform(data, lower=0.3, upper=1.7):
    """
    Transform hue
    Parameters: lower, upper
    """
    data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
    data = data.astype(np.float32)
    delta = random.uniform(lower, upper)
    data[1] *= delta
    data[1][data[1]>255] = 255
    data[1][data[1]<0] = 0
    data = data.astype(np.uint8)
    data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
    return data
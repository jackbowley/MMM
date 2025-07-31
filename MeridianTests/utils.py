# flake8: noqa: E501
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def adstock(x, rate):
    """
    Apply adstock transformation to a media variable.
    x: array-like, media spend or GRP
    rate: float, carryover rate between 0 and 1
    Returns: numpy array of adstocked values
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            result[i] = x[i]
        else:
            result[i] = x[i] + rate * result[i-1]
    return result
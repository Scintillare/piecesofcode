# Data science tools
import numpy as np

# Visualizations
import matplotlib.pyplot as plt

import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def show_transforms(data_dir, class_number, img_number, transforms):
    class_path = os.path.join(data_dir, os.listdir(data_dir)[class_number])
    img_path = os.path.join(class_path, os.listdir(class_path)[img_number])
    ex_img = Image.open(img_path)
    imshow(ex_img)

    plt.figure(figsize=(24, 24))

    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        _ = imshow_tensor(transforms(ex_img), ax=ax)

    plt.tight_layout()
    plt.show()

def plot_images(images, cls_true, label_names, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    X = images.numpy().transpose([0, 2, 3, 1])
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(np.clip(X[i, :, :, :], 0, 1), interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
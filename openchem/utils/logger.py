# Code referenced from
# https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# Updated for TensorFlow 2.x and modern Python 3
import io

import numpy as np
import tensorflow as tf
from PIL import Image


class Logger:

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_tensors = []
        for img in images:
            s = io.BytesIO()
            Image.fromarray(img).save(s, format="png")
            s.seek(0)
            img_tensor = tf.image.decode_png(s.read(), channels=4)
            img_tensors.append(img_tensor)

        img_tensors = tf.stack(img_tensors)
        with self.writer.as_default():
            tf.summary.image(tag, img_tensors, step=step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
        self.writer.flush()

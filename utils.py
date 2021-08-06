from shutil import get_terminal_size
import numpy as np
import sys
import time
import yaml
import logging


def load_yaml(load_path):
    with open(load_path, "r") as f:
        file = yaml.load(f, Loader=yaml.Loader)

    return file


def cant_finish_with_bar(path):
    if path[-1] == "/":
        path.pop(-1)
    return path


def must_finish_with_bar(path):
    if path[-1] != "/":
        path += "/"
    return path


def check_pixels(image, max=255, min=0):
    if (np.amax(image) > max) and (np.amin(image) < min):
        raise Exception(
            "Valor do pixels utrapassou o limite do intervalo [0, 255]."
            + f"Valor mínimo encontrado {np.amin(image)}, valor máximo encontrado {np.amax(image)}"
        )


def normalize(image):
    new_min, new_max = -1, 1
    min_value, max_value = 0, 255
    scale = new_max - new_min

    output = (scale * ((image - min_value) / (max_value - min_value))) + new_min
    check_pixels(output, max=1, min=-1)

    return output


def denormalize(image):
    """
    Args:
        image (np.array): Imagem normalizada em formato de array

    Returns:
        np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
    """
    min_value, max_value = -1, 1
    new_min, new_max = 0, 255
    scale = max_value - min_value

    image = image.numpy().clip(min_value, max_value)
    output = ((image - min_value) / scale) * (new_max - new_min)
    check_pixels(output)

    return output


class ProgressBar(object):
    """A progress bar which can print the progress modified from
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""

    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):

        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info(
                "terminal width is too small ({}), please consider "
                "widen the terminal for better progressbar "
                "visualization".format(terminal_width)
            )
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0

    def update(self, inf_str=""):
        """update"""
        self.completed += 1
        if not self.warm_up:
            self.start_time = time.time() - 1e-2
            self.warm_up = True
        elapsed = time.time() - self.start_time
        fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = "=" * mark_width + " " * (self.bar_width - mark_width)

        stdout_str = (
            "\rTraining [{}] {}/{}, {}  {:.1f} step/sec, time: {:02d}h {:02d}m {:02d}s"
        )
        seconds = int(elapsed % 60)
        minutes = int(elapsed // 60)
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        sys.stdout.write(
            stdout_str.format(
                bar_chars,
                self.completed,
                self.task_num,
                inf_str,
                fps,
                hours,
                minutes,
                seconds,
            )
        )

        sys.stdout.flush()

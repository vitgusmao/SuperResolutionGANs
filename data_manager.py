import glob
import ipdb
import math
import numpy as np
import os
import shutil
import tensorflow as tf

from PIL import Image, ImageEnhance, ImageOps

from utils import (
    must_finish_with_bar,
    cant_finish_with_bar,
    check_pixels,
    normalize,
    denormalize,
)


class ImagesManager:
    def __init__(self, config):
        img_info = config["images"]

        gt_size = img_info["gt_size"]
        lr_size = int(gt_size / img_info["scale"])
        self.lr_shape = (lr_size, lr_size)
        self.hr_shape = (gt_size, gt_size)

        datasets_info = config["datasets"]

        train_datasets = datasets_info["train_datasets"]
        self.train_dataset_names = train_datasets.keys()
        self.train_dataset_paths = [
            must_finish_with_bar(items["path"]) for items in train_datasets.values()
        ]

        test_datasets = datasets_info["test_datasets"]
        self.test_dataset_names = test_datasets.keys()
        self.test_dataset_paths = [
            must_finish_with_bar(items["path"]) for items in test_datasets.values()
        ]
        self.test_size = datasets_info["test_size"]

        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

        self.net_name = config["net"]
        self.base_monitor_path = "monitor/"
        self.train_monitor_paths = [
            f"{self.base_monitor_path}{dataset}/{self.net_name}/"
            for dataset in self.train_dataset_names
        ]
        self.format = "png"

        # Listando os nomes de todos os arquivos no diretório do dataset de treino
        self.train_images_names = [
            glob.glob("{}*.*".format(path)) for path in self.train_dataset_paths
        ]

        # Listando os nomes de todos os arquivos no diretório do dataset de treino
        self.test_images_names = [
            glob.glob("{}*.*".format(path)) for path in self.test_dataset_paths
        ][: self.test_size]
        [names.sort() for names in self.test_images_names]

    def process_image(self, image):
        image = np.array(image).astype(np.float32)
        check_pixels(image)
        return image

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image

    def load_image_batch(self, batch_size, images_paths, is_test=False):
        images_batch = []
        for i in range(batch_size):
            if is_test:
                image_name = images_paths[i]

                images_batch.append(self.load_image(image_name))

        return images_batch

    def resampling(self, image, shape):
        return image.resize(shape, resample=Image.BICUBIC)

    def augment_base(self, image):

        if np.random.uniform() < 0.5:
            image = ImageOps.mirror(image)

        if np.random.uniform() < 0.5:
            image = ImageOps.flip(image)

        return image

    def augment_x(self, image):

        # if np.random.uniform() < 0.5:
        #     bright_enhancer = ImageEnhance.Brightness(image)
        #     factor = np.random.uniform(0.5, 1.5)
        #     image = bright_enhancer.enhance(factor)

        # # Faz um ajuste randômico no contraste da imagem
        # if np.random.uniform() < 0.5:
        #     contrast_enhancer = ImageEnhance.Contrast(image)
        #     factor = np.random.uniform(0.5, 2.5)
        #     image = contrast_enhancer.enhance(factor)

        return image

    def get_images(self, batch_size, images_path, is_test=False):
        images = self.load_image_batch(batch_size, images_path, is_test)

        lr_images = []
        hr_images = []

        for image in images:

            lr_img = self.resampling(image, self.lr_shape)
            hr_img = self.resampling(image, self.hr_shape)

            lr_img = self.process_image(lr_img)
            hr_img = self.process_image(hr_img)

            # Faz a normalização da escala [0-255] para [0-1]
            lr_img = normalize(lr_img)
            hr_img = normalize(hr_img)

            lr_images.append(lr_img)
            hr_images.append(hr_img)

        # Transforma o array em tf.float32, o tipo de float que o tensorflow utiliza durante os cálculos
        lr_images = tf.cast(lr_images, dtype=tf.float32)
        hr_images = tf.cast(hr_images, dtype=tf.float32)

        return hr_images, lr_images

    def get_images_cnn(self, batch_size, images_path, is_test=False):
        images = self.load_image_batch(batch_size, images_path, is_test)

        lr_images = []
        hr_images = []

        for image in images:

            lr_img = self.resampling(image, self.lr_shape)
            lr_img = self.resampling(lr_img, self.hr_shape)
            hr_img = self.resampling(image, self.hr_shape)

            lr_img = self.process_image(lr_img)
            hr_img = self.process_image(hr_img)

            # Faz a normalização da escala [0-255] para [0-1]
            lr_img = normalize(lr_img)
            hr_img = normalize(hr_img)

            lr_images.append(lr_img)
            hr_images.append(hr_img)

        # Transforma o array em tf.float32, o tipo de float que o tensorflow utiliza durante os cálculos
        lr_images = tf.cast(lr_images, dtype=tf.float32)
        hr_images = tf.cast(hr_images, dtype=tf.float32)

        return hr_images, lr_images

    def unprocess_image(self, image, generated=False):
        image = np.array(image)
        image = image.astype(np.uint8)
        check_pixels(image)

        pil_image = Image.fromarray(image, "RGB")
        return pil_image

    def rebuild_images(self, images, generated=False):
        images = [denormalize(image) for image in images]
        images = [self.unprocess_image(image, generated) for image in images]

        return images

    def generate_and_save_images(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        for monitor_path, images_path in zip(
            self.train_monitor_paths, self.test_images_names
        ):

            images_names = f"{monitor_path}test_"

            _, lr_imgs = self.get_images(batch_size, images_path, is_test=True)

            hr_fakes = generator_net(lr_imgs, training=False)

            hr_fakes = self.rebuild_images(hr_fakes, generated=True)

            if not self.epochs:
                raise Exception("missing epochs")

            epoch = str(epoch)
            epoch = epoch.zfill(len(str(self.epochs)))

            for index, hr_gen in enumerate(hr_fakes):
                image_path = images_names + f"{index}/{epoch}.jpg"
                hr_gen.save(image_path)

    def generate_and_save_images_cnn(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        for monitor_path, images_path in zip(
            self.train_monitor_paths, self.test_images_names
        ):
            images_names = f"{monitor_path}test_"

            _, lr_imgs = self.get_images_cnn(batch_size, images_path, is_test=True)

            hr_fakes = generator_net(lr_imgs, training=False)

            hr_fakes = self.rebuild_images(hr_fakes, True)

            if not self.epochs:
                raise Exception("missing epochs")

            epoch = str(epoch)
            epoch = epoch.zfill(len(str(self.epochs)))

            for index, hr_gen in enumerate(hr_fakes):
                image_path = images_names + f"{index}/{epoch}_generated.jpg"
                hr_gen.save(image_path)

    def sample_interpolation(self, interpolation):

        lr_imgs = self.load_test_images()

        hr_interpolated = interpolation(images=lr_imgs)

        hr_interpolated = self.rebuild_images(hr_interpolated, True)

        if not self.epochs:
            raise Exception("missing epochs")

        for index, hr_gen in enumerate(hr_interpolated):
            name = str(index).zfill(len(str(len(hr_interpolated))))
            image_path = f"{self.train_monitor_path}{name}_generated.{self.format}"
            hr_gen.save(image_path)

    def initialize_dirs(self, testing_batch_size, total_epochs, originals=True):
        for monitor_path, images_path in zip(
            self.train_monitor_paths, self.test_images_names
        ):
            try:
                os.makedirs(monitor_path)
            except FileExistsError:
                shutil.rmtree(monitor_path, ignore_errors=True)
                os.makedirs(monitor_path)

            self.epochs = total_epochs
            imgs = self.load_image_batch(testing_batch_size, images_path, is_test=True)
            output_path = f"{monitor_path}test_"

            for idx, img in enumerate(imgs):
                sample_path = output_path + f"{idx}/"
                os.makedirs(sample_path, exist_ok=True)

                if originals:
                    hr_path = sample_path + f"high_resolution.{self.format}"
                    lr_path = sample_path + f"low_resolution.{self.format}"

                    hr_img = self.resampling(img, self.hr_shape)
                    hr_img = self.unprocess_image(hr_img)
                    hr_img.save(hr_path)

                    lr_img = self.resampling(img, self.lr_shape)
                    lr_img = self.unprocess_image(lr_img)
                    lr_img.save(lr_path)

    def get_dataset(self):
        opts = {
            "lr_shape": self.lr_shape,
            "hr_shape": self.hr_shape,
            "batch_size": self.batch_size,
        }
        train_images = []
        for images_sets in self.train_images_names:
            train_images.extend(images_sets)

        return InterpolatedImageLoader(train_images, opts)


def load_images_datasets(
    datasets_paths,
    batch_size,
    shuffle=True,
    buffer_size=10240,
):
    dataset = tf.data.Dataset.list_files(f"{datasets_paths[0]}*.*")
    for path in datasets_paths[1:]:
        dataset.concatenate(tf.data.Dataset.list_files(f"{path}*.*"))

    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # dataset = raw_dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_datasets(config, key, batch_size, shuffle=True, buffer_size=10240):
    datasets_config = config[key]
    datasets_paths = [items["path"] for items in datasets_config.values()]

    dataset = load_images_datasets(
        datasets_paths=datasets_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        buffer_size=buffer_size,
    )
    return dataset


def define_image_process(gt_size, scale):
    x_size = int(gt_size / scale)
    x_shape = (x_size, x_size)
    y_shape = (gt_size, gt_size)

    def process(images_paths):
        x_images, y_images = [], []

        for image_path in images_paths.numpy():
            image = Image.open(image_path)
            image = image.convert("RGB")

            if np.random.uniform() < 0.5:
                image = ImageOps.mirror(image)

            if np.random.uniform() < 0.5:
                image = ImageOps.flip(image)

            # if np.random.uniform() < 0.5:
            #     bright_enhancer = ImageEnhance.Brightness(image)
            #     factor = np.random.uniform(0.5, 1.5)
            #     x_image = bright_enhancer.enhance(factor)

            # # Faz um ajuste randômico no contraste da imagem
            # if np.random.uniform() < 0.5:
            #     contrast_enhancer = ImageEnhance.Contrast(x_image)
            #     factor = np.random.uniform(0.5, 2.5)
            #     x_image = contrast_enhancer.enhance(factor)

            x_image = image.resize(x_shape, resample=Image.BICUBIC)
            y_image = image.resize(y_shape, resample=Image.BICUBIC)

            x_image = np.array(x_image).astype(np.float32)
            y_image = np.array(y_image).astype(np.float32)

            x_image = normalize(x_image)
            y_image = normalize(y_image)

            x_images.append(x_image)
            y_images.append(y_image)

        return tf.cast(x_images, dtype=tf.float32), tf.cast(y_images, dtype=tf.float32)

    return process


def define_image_process_interpolated(gt_size, scale):
    x_size = int(gt_size / scale)
    x_shape = (x_size, x_size)
    y_shape = (gt_size, gt_size)

    def process(images_paths):
        x_images, y_images = [], []

        for image_path in images_paths.numpy():
            image = Image.open(image_path)
            image = image.convert("RGB")

            if np.random.uniform() < 0.5:
                image = ImageOps.mirror(image)

            if np.random.uniform() < 0.5:
                image = ImageOps.flip(image)

            # if np.random.uniform() < 0.5:
            #     bright_enhancer = ImageEnhance.Brightness(image)
            #     factor = np.random.uniform(0.5, 1.5)
            #     x_image = bright_enhancer.enhance(factor)

            # # Faz um ajuste randômico no contraste da imagem
            # if np.random.uniform() < 0.5:
            #     contrast_enhancer = ImageEnhance.Contrast(x_image)
            #     factor = np.random.uniform(0.5, 2.5)
            #     x_image = contrast_enhancer.enhance(factor)

            x_image = image.resize(x_shape, resample=Image.BICUBIC)
            x_image = image.resize(y_shape, resample=Image.BICUBIC)
            y_image = image.resize(y_shape, resample=Image.BICUBIC)

            x_image = np.array(x_image).astype(np.float32)
            y_image = np.array(y_image).astype(np.float32)

            x_image = normalize(x_image)
            y_image = normalize(y_image)

            x_images.append(x_image)
            y_images.append(y_image)

        return tf.cast(x_images, dtype=tf.float32), tf.cast(y_images, dtype=tf.float32)

    return process


class InterpolatedImageLoader(tf.keras.utils.Sequence):
    def __init__(self, images, opts):
        self.batch_size = opts.get("batch_size")
        self.hr_shape = opts.get("hr_shape")
        self.lr_shape = opts.get("lr_shape")
        self.images = images

    def __len__(self):
        return math.ceil(len(self.images) / (len(self.images) / self.batch_size))

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image

    def augment_base(self, image):

        if np.random.uniform() < 0.5:
            image = ImageOps.mirror(image)

        if np.random.uniform() < 0.5:
            image = ImageOps.flip(image)

        return image

    def process_base(self, image_path):
        img = self.load_image(image_path)
        img = self.augment_base(img)
        return img

    def resampling(self, image, shape):
        return image.resize(shape, resample=Image.BICUBIC)

    def convert_to_array(self, image):
        image = np.array(image).astype(np.float32)
        check_pixels(image)
        return image

    def process_x(self, image):
        image = self.resampling(image, self.lr_shape)
        image = self.resampling(image, self.hr_shape)
        image = self.convert_to_array(image)
        image = normalize(image)

        return image

    def process_y(self, image):
        image = self.resampling(image, self.hr_shape)
        image = self.convert_to_array(image)
        image = normalize(image)

        return image

    def __getitem__(self, idx):
        batch_x = self._get_x_batch(idx)
        batch_y = self._get_y_batch(idx)

        return np.array(batch_x), np.array(batch_y)

    def _get_image_batch(self, idx):
        imgs = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs = [self.process_base(img) for img in imgs]
        return imgs

    def _get_x_batch(self, idx):
        x = self._get_image_batch(idx)
        x = [self.process_x(img) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

    def _get_y_batch(self, idx):
        y = self._get_image_batch(idx)
        y = [self.process_y(img) for img in y]
        y = tf.cast(y, dtype=tf.float32)
        return y

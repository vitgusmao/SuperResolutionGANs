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
    def __init__(self, opts):
        img_info = opts.get("images")

        hr_size = img_info.get("hr_size")
        lr_size = img_info.get("lr_size")
        self.lr_shape = (lr_size, lr_size)
        self.hr_shape = (hr_size, hr_size)

        datasets_info = opts.get("datasets")

        train_dataset = datasets_info.get("train")
        self.train_dataset_name = train_dataset.get("name")
        self.train_dataset_dir = must_finish_with_bar(train_dataset.get("dir"))

        test_dataset = datasets_info.get("test")
        self.test_dataset_name = test_dataset.get("name")
        self.test_dataset_dir = must_finish_with_bar(test_dataset.get("dir"))
        self.test_size = test_dataset.get("size")

        self.batch_size = opts.get("batch_size")
        self.epochs = opts.get("epochs")

        self.net_name = opts.get("net")
        self.base_output_dir = "results/"
        self.train_monitor_log = (
            f"{self.base_output_dir}{self.train_dataset_name}/{self.net_name}/"
        )
        self.format = "png"

        # Listando os nomes de todos os arquivos no diretório do dataset de treino
        self.train_images_names = glob.glob("{}*.*".format(self.train_dataset_dir))
        np.random.shuffle(self.train_images_names)

        # Listando os nomes de todos os arquivos no diretório do dataset de treino
        self.test_images_names = glob.glob("{}*.*".format(self.test_dataset_dir))[
            : self.test_size
        ]
        np.random.shuffle(self.test_images_names)

    def process_image(self, image):
        image = np.array(image).astype(np.float32)
        check_pixels(image)
        return image

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image

    def load_image_batch(self, batch_size, is_test=False):
        images_batch = []
        for i in range(batch_size):
            if is_test:
                image_name = self.test_images_names[i]

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

    def get_images(self, batch_size, is_test=False):
        images = self.load_image_batch(batch_size, is_test)

        lr_images = []
        hr_images = []

        for image in images:
            lr_img = hr_img = image
            if not is_test:
                lr_img = hr_img = self.augment_base(image)
                lr_img = self.augment_x(image)

            lr_img = self.resampling(lr_img, self.lr_shape)
            hr_img = self.resampling(hr_img, self.hr_shape)

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

    def get_images_cnn(self, batch_size, path=None, is_test=False):
        images = self.load_raw_images(batch_size, path, is_test)

        lr_images = []
        hr_images = []

        if not is_test:
            images = self.augment(images)

        for image in images:

            lr_img = self.resampling(image, self.lr_shape)
            lr_img = self.resampling(lr_img, self.hr_shape)
            hr_img = self.resampling(image, self.hr_shape)

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
        # image = np.clip(image, 0, 255)

        # Se for uma imagem gerada, fazer uma correção de gamma
        if generated:
            image = 255 * ((image / 255) ** (2.2))
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
        images_names = f"{self.train_monitor_log}test_"

        _, lr_imgs = self.get_images(batch_size, is_test=True)

        hr_fakes = generator_net(lr_imgs, training=False)

        hr_fakes = self.rebuild_images(hr_fakes, generated=True)

        if not self.epochs:
            raise Exception("missing epochs")

        epoch = str(epoch)
        epoch = epoch.zfill(len(str(self.epochs)))

        for index, hr_gen in enumerate(hr_fakes):
            image_path = images_names + f"{index}/{epoch}_generated.jpg"
            hr_gen.save(image_path)

    def generate_and_save_images_cnn(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        images_names = f"{self.train_monitor_log}test_"

        _, lr_imgs = self.get_images_cnn(batch_size, is_test=True)

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
            image_path = f"{self.train_monitor_log}{name}_generated.{self.format}"
            hr_gen.save(image_path)

    def initialize_dirs(self, testing_batch_size, total_epochs, originals=True):
        try:
            os.makedirs(self.train_monitor_log)
        except FileExistsError:
            shutil.rmtree(self.train_monitor_log, ignore_errors=True)
            os.makedirs(self.train_monitor_log)

        self.epochs = total_epochs
        imgs = self.load_image_batch(testing_batch_size, is_test=True)
        images_dir = f"{self.train_monitor_log}test_"

        for idx, img in enumerate(imgs):
            sample_dir = images_dir + f"{idx}/"
            os.makedirs(sample_dir, exist_ok=True)

            if originals:
                hr_path = sample_dir + f"high_resolution.{self.format}"
                lr_path = sample_dir + f"low_resolution.{self.format}"

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

        return ImageLoader(self.train_images_names, opts)


class ImageLoader(tf.keras.utils.Sequence):
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
        # if np.random.uniform() < 0.5:
        #     bright_enhancer = ImageEnhance.Brightness(image)
        #     factor = np.random.uniform(0.5, 1.5)
        #     image = bright_enhancer.enhance(factor)

        # # Faz um ajuste randômico no contraste da imagem
        # if np.random.uniform() < 0.5:
        #     contrast_enhancer = ImageEnhance.Contrast(image)
        #     factor = np.random.uniform(0.5, 2.5)
        #     image = contrast_enhancer.enhance(factor)

        image = self.resampling(image, self.lr_shape)
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


class InterpolatedImageLoader(ImageLoader):
    def process_x(self, image):
        # if np.random.uniform() < 0.5:
        #     bright_enhancer = ImageEnhance.Brightness(image)
        #     factor = np.random.uniform(0.5, 1.5)
        #     image = bright_enhancer.enhance(factor)

        # # Faz um ajuste randômico no contraste da imagem
        # if np.random.uniform() < 0.5:
        #     contrast_enhancer = ImageEnhance.Contrast(image)
        #     factor = np.random.uniform(0.5, 2.5)
        #     image = contrast_enhancer.enhance(factor)

        image = self.resampling(image, self.lr_shape)
        image = self.resampling(image, self.hr_shape)
        image = self.convert_to_array(image)
        image = normalize(image)

        return image


def run_interpolations(dataset_info, img_shapes):
    hr_img_shape = img_shapes.get("hr_img_shape")
    lr_img_shape = img_shapes.get("lr_img_shape")

    hr_shape = img_shapes.get("hr_shape")
    reshape_size = hr_shape[0]

    dataset_dir = dataset_info.get("dataset_dir")
    dataset_name = dataset_info.get("dataset_name")

    methods = ["nearest", "linear", "area", "cubic"]

    for method in methods:
        image_manager = ImagesManager(
            dataset_dir, dataset_name, method, hr_img_shape, lr_img_shape
        )

        seq = iaa.Sequential([iaa.Resize(reshape_size, interpolation=method)])

        image_manager.sample_interpolation(seq)

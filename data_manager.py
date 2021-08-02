import os
import shutil
import glob
import ipdb
import math
import numpy as np
from numpy.lib.twodim_base import flipud
import tensorflow as tf

from PIL import Image, ImageEnhance, ImageOps


def check_pixels(image, max=255, min=0):
    if (np.amax(image) > max) and (np.amin(image) < min):
        raise Exception(
            "Valor do pixels utrapassou o limite do intervalo [0, 255]."
            + f"Valor mínimo encontrado {np.amin(image)}, valor máximo encontrado {np.amax(image)}"
        )


def cant_finish_with_bar(path):
    if path[-1] == "/":
        path.pop(-1)
    return path


def must_finish_with_bar(path):
    if path[-1] != "/":
        path += "/"
    return path


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

    def normalize(self, input_data):
        new_min, new_max = -1, 1
        min_value, max_value = 0, 255
        scale = new_max - new_min

        normalized = (
            scale * ((input_data - min_value) / (max_value - min_value))
        ) + new_min
        check_pixels(normalized, max=1, min=0)
        return normalized

    def denormalize(self, input_data):
        """
        Args:
            input_data (np.array): Imagem normalizada em formato de array

        Returns:
            np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
        """
        min_value, max_value = -1, 1
        new_min, new_max = 0, 255
        scale = max_value - min_value

        output = ((input_data - min_value) / scale) * (new_max - new_min)

        check_pixels(output)

        return output

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

    def load_test_images(self):
        pass

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
            lr_img = self.normalize(lr_img)
            hr_img = self.normalize(hr_img)

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
            lr_img = self.normalize(lr_img)
            hr_img = self.normalize(hr_img)

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
        # if generated:
        #     image = 255 * ((image / 255) ** (2.2))
        image = image.astype(np.uint8)
        check_pixels(image)
        pil_image = Image.fromarray(image, "RGB")

        return pil_image

    def rebuild_images(self, images, generated=False):
        images = [self.denormalize(image) for image in images]
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

        hr_fakes = self.rebuild_images(hr_fakes, True)

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
            image_path = f"{self.train_monitor_log}{name}_generated.jpg"
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
                hr_path = sample_dir + "high_resolution.jpg"
                lr_path = sample_dir + "low_resolution.jpg"

                hr_img = self.resampling(img, self.hr_shape)
                hr_img = self.unprocess_image(hr_img)
                hr_img.save(hr_path)

                lr_img = self.resampling(img, self.lr_shape)
                lr_img = self.unprocess_image(lr_img)
                lr_img.save(lr_path)


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, image_manager, batch_size):
        self.mngr = image_manager

        self.images = self.mngr.train_images_names
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / (len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self._get_x_batch(idx)
        batch_y = self._get_y_batch(idx)

        return np.array(batch_x), np.array(batch_y)

    def _get_image_batch(self, idx):
        imgs = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs = [self.mngr.load_image(img) for img in imgs]
        imgs = [self.mngr.augment_base(img) for img in imgs]
        return imgs

    def _get_x_batch(self, idx):
        x = self._get_image_batch(idx)
        x = [self.mngr.augment_x(img) for img in x]
        x = [self.mngr.resampling(img, self.mngr.lr_shape) for img in x]
        x = [self.mngr.process_image(img) for img in x]
        x = [self.mngr.normalize(img) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

    def _get_y_batch(self, idx):
        y = self._get_image_batch(idx)
        y = [self.mngr.resampling(img, self.mngr.hr_shape) for img in y]
        y = [self.mngr.process_image(img) for img in y]
        y = [self.mngr.normalize(img) for img in y]
        y = tf.cast(y, dtype=tf.float32)
        return y


class CNNImageSequence(ImageSequence):
    def _get_x_batch(self, idx):
        x = self._get_image_batch(idx)
        x = [self.mngr.augment_x(img) for img in x]
        x = [self.mngr.resampling(img, self.mngr.lr_shape) for img in x]
        x = [self.mngr.resampling(img, self.mngr.hr_shape) for img in x]
        x = [self.mngr.process_image(img) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x


def get_image_dataset(images_manager, batch_size):
    images_manager
    images_dataset = tf.data.Dataset.list_files("../datasets/DIV2K_train_HR/*.*")

    def process_path(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=3)
        # img = img.numpy()
        # img = images_manager.augment_base(img)

        # img = images_manager.augment_x(img)

        # x_images = images_manager.resampling(img, images_manager.lr_shape)
        # y_images = images_manager.resampling(img, images_manager.hr_shape)
        x_images = tf.image.resize(img, images_manager.lr_shape)
        y_images = tf.image.resize(img, images_manager.hr_shape)

        return x_images, y_images

    images_dataset = images_dataset.map(
        process_path, num_parallel_calls=tf.data.AUTOTUNE
    )

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    images_dataset = configure_for_performance(images_dataset)

    return images_dataset


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

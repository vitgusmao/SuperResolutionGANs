import os
import glob
import ipdb
import math
import numpy as np
from numpy.lib.twodim_base import flipud
import tensorflow as tf
import imgaug.augmenters as iaa
import imgaug

from PIL import Image


class ImagesManager:
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        net_name,
        hr_shape,
        lr_shape,
        output_format="PNG",
    ):

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir.format(dataset_name)

        self.lr_shape = lr_shape
        self.hr_shape = hr_shape

        self.net_name = net_name
        self.base_output_dir = "results/"
        self.output_loc = f"{self.base_output_dir}{self.dataset_name}/{self.net_name}/"
        self.output_format = output_format

        # Listando os nomes de todos os arquivos no diretório do dataset
        self.images_names = glob.glob("{}*.*".format(self.dataset_dir))

        # Copiando nome das imagens para treino e randomizando sua disposição
        self.train_images = self.images_names.copy()
        np.random.shuffle(self.train_images)

        # Copiando nome das imagens para teste e randomizando sua disposição
        self.test_images = self.images_names.copy()[: int(len(self.images_names) / 10)]
        np.random.shuffle(self.test_images)

    def normalize(self, input_data, scale_map=(0, 1)):
        new_min, new_max = scale_map
        max_value = 255
        min_value = 0
        scale = new_max - new_min

        return (scale * ((input_data - min_value) / (max_value - min_value))) + new_min

    def denormalize(self, input_data, scale_map=(0, 1)):
        """
        Args:
            input_data (np.array): Imagem normalizada em formato de array

        Returns:
            np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
        """
        min_value, max_value = scale_map
        new_max = 255
        new_min = 0
        scale = max_value - min_value

        input_data = ((input_data - min_value) / scale) * (new_max - new_min)

        if (np.amax(input_data) > 255) and (np.amin(input_data) < 0):
            raise Exception(
                f"Valor do pixels utrapassou o limite do intervalo [0, 255]. Valor mínimo encontrado {np.amin(input_data)}, valor máximo encontrado {np.amax(input_data)}"
            )

        return input_data

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32)
        return image

    def load_images(self, batch_size, path=None, is_test=False):
        images_batch = []
        for i in range(batch_size):
            if not path:
                if not is_test:
                    # image_name = self.train_images.pop(i)
                    image_name = self.train_images[i]
                else:
                    image_name = self.test_images[i]

                images_batch.append(self.load_image(image_name))

            else:
                images_batch.append(self.load_image(path))

        return images_batch

    def resampling(self, image, shape):
        """Redimensiona a imagem para a resolução desejada

        Args:
            image ([type]): [description]
            shape ([type]): [description]
        """
        seq = iaa.Sequential([iaa.Resize(shape[0], interpolation="cubic")])

        return seq(image=image)

    def augment(self, images):
        if isinstance(images, list):
            images = np.array(images).astype(np.uint8)

        org_img_size = images.shape[1]

        seq = iaa.Sequential(
            [
                iaa.Crop(px=(0, int(org_img_size / 4)), keep_size=True),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.MultiplyBrightness((0.7, 1.3)),
                iaa.KeepSizeByResize(
                    iaa.Resize(np.random.uniform(0.75, 1.75), interpolation=imgaug.ALL)
                ),
                iaa.MultiplyHueAndSaturation(1.5),
            ]
        )

        images = seq(images=images)

        return images

    def load_test_images(self):
        pass

    def get_images(self, batch_size, path=None, is_test=False):
        images = self.load_images(batch_size, path, is_test)

        lr_images = []
        hr_images = []

        if not is_test:
            images = self.augment(images)

        for image in images:

            lr_img = self.resampling(image, self.lr_shape)
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
        image = np.clip(image, 0, 255)

        # Se for uma imagem gerada, fazer uma correção de gamma
        # if generated:
        #     image = 255 * ((image / 255) ** (2.2))

        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image, "RGB")

        return pil_image

    def rebuild_images(self, images, generated=False, scale_map=None):
        if not scale_map:
            if generated:
                scale_map = (-1, 1)
            else:
                scale_map = (0, 1)

        images = [self.denormalize(image, scale_map) for image in images]
        images = [self.unprocess_image(image, generated) for image in images]

        return images

    def sample_per_epoch(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        images_names = f"{self.output_loc}test_"

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

    def sample_per_epoch_cnn(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        images_names = f"{self.output_loc}test_"

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

    def sample_specific(self, generator_net, image_path, image_name):
        img_format = self.output_format.lower()
        os.makedirs(image_path, exist_ok=True)

        original_path = image_path + image_name
        gen_path = image_path + f"test_gen.{img_format}"
        low_path = image_path + f"low_resolution.{img_format}"

        _, lr_img = self.get_images(1, original_path, True)

        hr_gen = generator_net.predict(lr_img)
        hr_gen = self.rebuild_images(hr_gen, True)[0]

        lr_img = self.rebuild_images(lr_img)[0]

        hr_gen.save(gen_path)
        lr_img.save(low_path)

    def initialize_dirs(self, testing_batch_size, total_epochs, originals=True):
        os.makedirs(f"{self.output_loc}", exist_ok=True)

        self.epochs = total_epochs
        imgs = self.load_images(testing_batch_size, is_test=True)
        images_dir = f"{self.output_loc}test_"

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
        self.data_manager = image_manager

        self.images = self.data_manager.images_names
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / (len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self._get_x_batch(idx)
        batch_y = self._get_y_batch(idx)

        return np.array(batch_x), np.array(batch_y)

    def _get_image_batch(self, idx):
        imgs = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs = [self.data_manager.load_image(img) for img in imgs]
        imgs = self.data_manager.augment(imgs)
        return imgs

    def _get_x_batch(self, idx):
        x = self._get_image_batch(idx)
        x = [self.data_manager.resampling(img, self.data_manager.lr_shape) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

    def _get_y_batch(self, idx):
        y = self._get_image_batch(idx)
        y = [self.data_manager.resampling(img, self.data_manager.hr_shape) for img in y]
        y = tf.cast(y, dtype=tf.float32)
        return y


class CNNImageSequence(ImageSequence):
    def _get_x_batch(self, idx):
        x = self._get_image_batch(idx)
        x = [self.data_manager.resampling(img, self.data_manager.lr_shape) for img in x]
        x = [self.data_manager.resampling(img, self.data_manager.hr_shape) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

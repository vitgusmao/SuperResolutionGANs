import os
import glob
import ipdb
import math
import numpy as np
import tensorflow as tf

from PIL import Image, ImageEnhance


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

    def load_raw_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        return image

    def load_raw_images(self, batch_size, path=None, is_test=False):
        images_batch = []
        for i in range(batch_size):
            if not path:
                if not is_test:
                    image_name = self.train_images.pop(i)
                    image_name = self.train_images[i]
                else:
                    image_name = self.test_images[i]

                images_batch.append(self.load_raw_image(image_name))

            else:
                images_batch.append(self.load_raw_image(path))

        return images_batch

    def upsampling(self, pil_image, shape):
        """Redimensiona a imagem para as dimensões de alta resolução

        Args:
            pil_image ([type]): [description]
            shape ([type]): [description]
        """
        return pil_image.resize(shape, resample=Image.BICUBIC)

    def downsampling(self, pil_image, shape):
        """Redimensiona a imagem para as dimensões de baixa resolução
            utilizando a técnica Lanczos que é um filtro de alta qualidade
            para downsampling de imagens

        Args:
            pil_image ([type]): [description]
            shape ([type]): [description]
        """
        return pil_image.resize(shape, resample=Image.LANCZOS)

    def train_augmentation(self, image):
        interpolation_methods = [
            "bilinear",
            "nearest",
            "bicubic",
            "gaussian",
            "mitchellcubic",
        ]

        # if np.random.uniform() < 0.5:
        #     image = tf.image.random_flip_left_right(image)
        # if np.random.uniform() < 0.5:
        #     image = tf.image.random_flip_up_down(image)
        # if np.random.uniform() < 0.8:
        #     image = tf.image.random_brightness(image, max_delta=2.0)
        # if np.random.uniform() < 0.9:
        #     image = tf.image.random_saturation(image, lower=0.5, upper=5.0)
        # if np.random.uniform() < 0.6:
        #     original_size = image.shape[0]
        #     crop_size = np.random.randint(int(image.shape[0] * (3 / 4)), image.shape[0])
        #     image = tf.image.random_crop(image, size=(crop_size, crop_size, 3))
        #     image = tf.image.resize(
        #         image,
        #         size=(original_size, original_size),
        #         method=np.random.choice(interpolation_methods),
        #     )
        # if np.random.uniform() < 0.9:
        #     original_size = image.shape[0]
        #     scale_factor = np.random.choice([0.33, 0.5, 1, 2, 3])
        #     image = tf.image.resize(
        #         image,
        #         size=(
        #             int(original_size * scale_factor),
        #             int(original_size * scale_factor),
        #         ),
        #     )
        #     image = tf.image.resize(
        #         image,
        #         size=(original_size, original_size),
        #         method=np.random.choice(interpolation_methods),
        #     )

        return image

    def train_preprocess(self, pil_image):

        # Transforma a imagem em formato numpy.array
        image = np.array(pil_image).astype(np.float32)

        image = self.train_augmentation(image)

        # Faz a normalização da escala [0-255] para [0-1]
        image = self.normalize(image)

        return image

    def test_preprocess(self, pil_image):

        # Transforma a imagem em formato numpy.array do tipo np.float32
        output_image = np.array(pil_image).astype(np.float32)

        # Faz a normalização da escala [0-255] para [0-1]
        output_image = self.normalize(output_image)

        return output_image

    def load_images(self, batch_size, path=None, is_test=False):
        images = self.load_raw_images(batch_size, path, is_test)

        lr_images = []
        hr_images = []
        for image in images:
            # Pré-processamento das imagens
            hr_img = self.upsampling(image, self.hr_shape)
            lr_img = self.downsampling(image, self.lr_shape)

            if not is_test:
                hr_img = self.train_preprocess(hr_img)
                lr_img = self.train_preprocess(lr_img)

            else:
                hr_img = self.test_preprocess(hr_img)
                lr_img = self.test_preprocess(lr_img)

            hr_images.append(hr_img)
            lr_images.append(lr_img)

        # Transforma o array em tf.float32, o tipo de float que o tensorflow utiliza durante os cálculos
        hr_images = tf.cast(hr_images, dtype=tf.float32)
        lr_images = tf.cast(lr_images, dtype=tf.float32)

        return hr_images, lr_images

    def load_images_cnn(self, batch_size, path=None, is_test=False):
        images = self.load_raw_images(batch_size, path, is_test)

        lr_images = []
        hr_images = []
        for image in images:
            # Pré-processamento das imagens
            hr_img = self.upsampling(image, self.hr_shape)
            lr_img = self.downsampling(image, self.lr_shape)
            lr_img = self.upsampling(lr_img, self.hr_shape)

            if not is_test:
                hr_img = self.train_preprocess(hr_img)
                lr_img = self.train_preprocess(lr_img)

            else:
                hr_img = self.test_preprocess(hr_img)
                lr_img = self.test_preprocess(lr_img)

            hr_images.append(hr_img)
            lr_images.append(lr_img)

        # Transforma o array em tf.float32, o tipo de float que o tensorflow utiliza durante os cálculos
        hr_images = tf.cast(hr_images, dtype=tf.float32)
        lr_images = tf.cast(lr_images, dtype=tf.float32)

        return hr_images, lr_images

    def unprocess_image(self, image, scale_map, generated):
        image = np.array(image)
        image = self.denormalize(image, scale_map)

        # Se for uma imagem gerada, fazer uma correção de gamma
        # if generated:
        #     image = 255 * ((image / 255) ** (2.2))

        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image, "RGB")

        return pil_image

    def rebuild_images(self, images, generated=False, scale_map=None):
        output_images = []
        if not scale_map:
            if generated:
                scale_map = (-1, 1)
            else:
                scale_map = (0, 1)

        output_images = [
            self.unprocess_image(image, scale_map, generated) for image in images
        ]

        return output_images

    def sample_per_epoch(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        images_names = f"{self.output_loc}test_"

        _, lr_imgs = self.load_images(batch_size, is_test=True)

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

        _, lr_imgs = self.load_images_cnn(batch_size, is_test=True)

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

        _, lr_img = self.load_images(1, original_path, True)

        hr_gen = generator_net.predict(lr_img)
        hr_gen = self.rebuild_images(hr_gen, True)[0]

        lr_img = self.rebuild_images(lr_img)[0]

        hr_gen.save(gen_path)
        lr_img.save(low_path)

    def initialize_dirs(self, testing_batch_size, total_epochs):
        os.makedirs(f"{self.output_loc}", exist_ok=True)

        self.epochs = total_epochs
        imgs = self.load_raw_images(testing_batch_size, is_test=True)
        images_dir = f"{self.output_loc}test_"

        for idx, img in enumerate(imgs):
            sample_dir = images_dir + f"{idx}/"
            os.makedirs(sample_dir, exist_ok=True)
            hr_path = sample_dir + "?0_high_resolution.jpg"
            lr_path = sample_dir + "?0_low_resolution.jpg"

            hr_img = self.upsampling(img, self.hr_shape)
            lr_img = self.downsampling(img, self.lr_shape)

            hr_img.save(hr_path)
            lr_img.save(lr_path)


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, image_manager, batch_size):
        self.data_manager = image_manager

        self.x, self.y = self.data_manager.images_names, self.data_manager.images_names
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / (len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self._get_x_batch(idx)
        batch_y = self._get_y_batch(idx)

        return np.array(batch_x), np.array(batch_y)

    def _get_x_batch(self, idx):
        x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = [self.data_manager.load_raw_image(img) for img in x]
        x = [
            self.data_manager.downsampling(img, self.data_manager.lr_shape)
            for img in x
        ]
        x = [self.data_manager.train_preprocess(img) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

    def _get_y_batch(self, idx):
        y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        y = [self.data_manager.load_raw_image(img) for img in y]
        y = [
            self.data_manager.upsampling(img, self.data_manager.hr_shape) for img in y
        ]
        y = [self.data_manager.train_preprocess(img) for img in y]
        y = tf.cast(y, dtype=tf.float32)
        return y


class CNNImageSequence(ImageSequence):
    def _get_x_batch(self, idx):
        x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = [self.data_manager.load_raw_image(img) for img in x]
        x = [
            self.data_manager.downsampling(img, self.data_manager.lr_shape)
            for img in x
        ]
        x = [
            self.data_manager.upsampling(img, self.data_manager.hr_shape) for img in x
        ]
        x = [self.data_manager.train_preprocess(img) for img in x]
        x = tf.cast(x, dtype=tf.float32)
        return x

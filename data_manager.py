import os
import glob
import ipdb
import numpy as np
import tensorflow as tf

from PIL import Image, ImageEnhance
from imageio import imwrite


def normalize(input_data, scale_map=(0, 1)):
    new_min, new_max = scale_map
    max_value = 255
    min_value = 0
    scale = new_max - new_min

    return (scale * ((input_data - min_value) /
                     (max_value - min_value))) + new_min


def denormalize(input_data, scale_map=(0, 1)):
    """
    Args:
        input_data (np.array): Imagem normalizada em formato de array

    Returns:
        np.array: Imagem desnormalizada com pixels de [0 - 255] no formato uint8
    """
    max_value, min_value = scale_map
    new_max = 255
    new_min = 0
    scale = max_value - min_value

    input_data = ((input_data - min_value) / scale) * (new_max - new_min)

    if (np.amax(input_data) > 255) and (np.amin(input_data) < 0):
        raise ValueError

    return input_data


class ImagesManager:
    def __init__(self,
                 dataset_dir=None,
                 dataset_name=None,
                 hr_shape=None,
                 lr_shape=None):
        if dataset_dir and dataset_name and hr_shape and lr_shape:
            self.dataset_name = dataset_name
            self.dataset_dir = dataset_dir.format(dataset_name)
            self.lr_shape = lr_shape
            self.hr_shape = hr_shape

            # Listando os nomes de todos os arquivos no diretório do dataset
            images_names = glob.glob('{}*.*'.format(self.dataset_dir))

            # Copiando nome das imagens para treino e randomizando sua disposição
            self.train_images = images_names.copy()
            np.random.shuffle(self.train_images)

            # Copiando nome das imagens para teste e randomizando sua disposição
            self.test_images = images_names.copy()[:1000]
            np.random.shuffle(self.test_images)

    def _load_raw_image(self, image_path):
        image = Image.open(image_path)
        image.convert('RGB')
        return image

    def _load_images(self, batch_size, path=None, is_test=False):
        images_batch = []
        for i in range(batch_size):
            if not path:
                if not is_test:
                    image_name = self.train_images.pop(i)
                else:
                    image_name = self.test_images[i]

                images_batch.append(self._load_raw_image(image_name))

            else:
                images_batch.append(self._load_raw_image(path))

        return images_batch

    def _upsampling(self, pil_image, shape):
        """Redimensiona a imagem para as dimensões de alta resolução

        Args:
            pil_image ([type]): [description]
            shape ([type]): [description]
        """
        return pil_image.resize(shape, resample=Image.BICUBIC)

    def _downsampling(self, pil_image, shape):
        """Redimensiona a imagem para as dimensões de baixa resolução
            utilizando a técnica Lanczos que é um filtro de alta qualidade
            para downsampling de imagens

        Args:
            pil_image ([type]): [description]
            shape ([type]): [description]
        """
        return pil_image.resize(shape, resample=Image.LANCZOS)

    def _train_preprocess(self, pil_image):

        # # Faz um flip da imagem randomicamente
        # pil_image = tf.image.random_flip_left_right(pil_image)

        # Faz um ajuste randômico no brilho da imagem
        bright_enhancer = ImageEnhance.Brightness(pil_image)
        factor = 0.5
        pil_image = bright_enhancer.enhance(factor)

        # Faz um ajuste randômico no contraste da imagem
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        factor = 4.0
        pil_image = contrast_enhancer.enhance(factor)

        # # Faz um ajuste randômico na saturação da imagem
        # sat_enhancer = ImageEnhance.Color(pil_image)
        # factor = 1.5
        # pil_image = sat_enhancer.enhance(factor)

        # Transforma a imagem em formato numpy.array
        output_image = np.array(pil_image).astype(np.float32)

        # Faz a normalização da escala [0-255] para [0-1]
        output_image = normalize(output_image)

        return output_image

    def _test_preprocess(self, pil_image):

        # Transforma a imagem em formato numpy.array do tipo np.float32
        output_image = np.array(pil_image).astype(np.float32)

        # Faz a normalização da escala [0-255] para [0-1]
        output_image = normalize(output_image)

        return output_image

    def load_images(self, batch_size, path=None, is_test=False):
        images = self._load_images(batch_size, path, is_test)

        lr_images = []
        hr_images = []
        for image in images:
            # Pré-processamento das imagens
            hr_img = self._upsampling(image, self.hr_shape)
            lr_img = self._downsampling(image, self.lr_shape)

            if not is_test:
                hr_img = self._train_preprocess(hr_img)
                lr_img = self._train_preprocess(lr_img)

            else:
                hr_img = self._test_preprocess(hr_img)
                lr_img = self._test_preprocess(lr_img)

            hr_images.append(hr_img)
            lr_images.append(lr_img)

        # Transforma o array em tf.float32, o tipo de float que o tensorflow utiliza durante os cálculos
        hr_images = tf.cast(hr_images, dtype=tf.float32)
        lr_images = tf.cast(lr_images, dtype=tf.float32)

        return hr_images, lr_images

    def _unprocess_default_image(self, image):
        image = np.array(image)
        image = denormalize(image)
        image.astype(np.uint8)
        pil_image = Image.fromarray(image, "RGB")

        return pil_image

    def _unprocess_image(self, image, scale_map):
        image = np.array(image)
        image = denormalize(image, scale_map)
        image.astype(np.uint8)
        pil_image = Image.fromarray(image, "RGB")

        return pil_image

    def rebuild_images(self, images, generated=False):
        output_images = []
        if generated:
            scale_map = (-1, 1)
        else:
            scale_map = (0, 1)

        output_images = [
            self._unprocess_image(image, scale_map) for image in images
        ]

        return output_images

    def sample_per_epoch(
        self,
        generator_net,
        epoch,
        batch_size,
    ):
        images_names = f'imgs/{self.dataset_name}/test_'

        _, lr_imgs = self.load_images(batch_size, is_test=True)

        hr_fakes = generator_net.predict(lr_imgs)

        hr_fakes = self.rebuild_images(hr_fakes, True)

        if not self.epochs:
            raise Exception('missing epochs')

        epoch = str(epoch)
        epoch = epoch.zfill(len(str(self.epochs)))

        for index, hr_gen in enumerate(hr_fakes):
            image_path = images_names + f'{index}/{epoch}_generated.jpg'
            hr_gen.save(image_path)

    # def sample_images(self, hr_images=None, fake_images=None):
    #     os.makedirs('imgs/{}'.format(self.dataset_name), exist_ok=True)

    #     for hr_img, (index,
    #                  hr_gen) in zip(hr_images,
    #                                 zip(range(len(fake_images)), fake_images)):

    #         imwrite(
    #             'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
    #                                        'test_gen_hr'), hr_gen)
    #         imwrite(
    #             'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
    #                                        'test_hr'), hr_img)

    def sample_specific(self, generator_net, image_path, image_name):
        os.makedirs(image_path, exist_ok=True)
        original_path = image_path + image_name
        gen_path = image_path + 'test_gen.jpg'
        low_path = image_path + 'low_resolution.jpg'

        hr_img, lr_img = self.load_images(1, original_path, True)

        hr_gen = generator_net.predict(lr_img)
        hr_gen = self.rebuild_images(hr_gen, True)

        lr_img = self.rebuild_images(lr_img)

        hr_gen.save(gen_path)
        hr_img.save(low_path)

    def initialize_dirs(self, testing_batch_size, total_epochs):
        os.makedirs(f'imgs/{self.dataset_name}', exist_ok=True)

        self.epochs = total_epochs
        imgs = self._load_images(testing_batch_size, is_test=True)
        images_dir = f'imgs/{self.dataset_name}/test_'

        for idx, img in enumerate(imgs):
            sample_dir = images_dir + f'{idx}/'
            os.makedirs(sample_dir, exist_ok=True)
            hr_path = sample_dir + '?0_high_resolution.jpg'
            lr_path = sample_dir + '?0_low_resolution.jpg'

            hr_img = self._upsampling(img, self.hr_shape)
            lr_img = self._downsampling(img, self.lr_shape)

            hr_img.save(hr_path)
            lr_img.save(lr_path)

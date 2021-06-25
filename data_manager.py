import os
import glob
import ipdb
import numpy as np

from imageio import imread, imwrite
from skimage.transform import resize

from utils import denormalize, normalize


class DataManager:
    def __init__(self, dataset_dir, dataset_name, hr_shape, lr_shape):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir.format(dataset_name)
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape

        # Listando os nomes de toso os arquivos no diretório do dataset
        self.all_train_images = glob.glob('{}*.*'.format(self.dataset_dir))
        self.all_test_images = glob.glob('{}*.*'.format(self.dataset_dir))

        # Gerando lista aleatória de índices de arquivos para testes
        self.test_random_indexes = np.random.randint(0,
                                                     len(self.all_test_images),
                                                     len(self.all_test_images))

    def load_data(self, batch_size, is_testing=False):

        # Choose a random batch of images
        if is_testing:
            all_images = self.all_test_images
            images_batch_indexes = self.test_random_indexes[:batch_size]
        else:
            all_images = self.all_train_images
            images_batch_indexes = np.random.randint(0, len(all_images),
                                                     batch_size)

        images_batch = []
        for i in images_batch_indexes:
            images_batch.append(all_images[i])
            if not is_testing:
                del all_images[i]

        return map(lambda img: imread(img, pilmode='RGB').astype(np.float32),
                   images_batch)

    def prepare_data(self, images):
        lr_images = []
        hr_images = []
        for image in images:
            # Resize the image
            high_resolution_img = resize(image, self.hr_shape)
            low_resolution_img = resize(image, self.lr_shape)

            hr_images.append(high_resolution_img)
            lr_images.append(low_resolution_img)

        return normalize(np.array(hr_images)), normalize(np.array(lr_images))

    def load_prepared_data(self, batch_size, is_testing=False):
        images = self.load_data(batch_size=batch_size, is_testing=is_testing)
        return self.prepare_data(images)

    def sample_images(self,
                      generator_net=None,
                      epoch=None,
                      batch_size=1,
                      hr_images=None,
                      fake_images=None):
        os.makedirs('imgs/%s' % self.dataset_name, exist_ok=True)

        if generator_net:
            _, lr_imgs = self.load_prepared_data(batch_size=batch_size,
                                                 is_testing=True)

            hr_fakes = generator_net.predict(lr_imgs)

            hr_fakes = denormalize(hr_fakes, min_value=-1)

            if not self.epochs:
                raise Exception('missing epochs')

            epoch = str(epoch)
            epoch = epoch.zfill(len(str(self.epochs)))

            for index, hr_gen in zip(range(len(hr_fakes)), hr_fakes):
                imwrite(
                    'imgs/{}/{}/{}_{}.jpg'.format(self.dataset_name, index,
                                                  epoch, 'generated'),
                    hr_gen.astype(np.uint8))

        elif hr_images is not None and fake_images is not None:
            for hr_img, (index, hr_gen) in zip(
                    hr_images, zip(range(len(fake_images)), fake_images)):

                imwrite(
                    'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
                                               'test_gen_hr'), hr_gen)
                imwrite(
                    'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
                                               'test_hr'), hr_img)

    def initialize_dirs(self, testing_batch_size, total_epochs):
        self.epochs = total_epochs
        hr_imgs, lr_imgs = self.load_prepared_data(
            batch_size=testing_batch_size, is_testing=True)

        hr_imgs = denormalize(hr_imgs)
        lr_imgs = denormalize(lr_imgs)

        for i in range(testing_batch_size):
            os.makedirs('imgs/{}/{}'.format(self.dataset_name, i),
                        exist_ok=True)

        for index, (hr_img, lr_img) in zip(range(len(hr_imgs)),
                                           zip(hr_imgs, lr_imgs)):
            imwrite(
                'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
                                           '?0_high_resolution'),
                hr_img.astype(np.uint8))
            imwrite(
                'imgs/{}/{}/{}.jpg'.format(self.dataset_name, index,
                                           '?0_low_resolution'),
                lr_img.astype(np.uint8))

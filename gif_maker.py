import imageio
import glob

dataset_name = 'img_align_celeba'

for samples in range(2):
    imgs_dir = 'imgs/{}/{}/'.format(dataset_name, samples)
    file_names = glob.glob('{}*.*'.format(imgs_dir))
    file_names.sort()

    images = []
    with imageio.get_writer('imgs/{}/{}.gif'.format(dataset_name, samples),
                            mode='I') as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)

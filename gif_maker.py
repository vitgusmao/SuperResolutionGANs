import imageio
import glob
import argparse

parser = argparse.ArgumentParser(description='Options for build the gif')
parser.add_argument('--input', help='path to images folder')
parser.add_argument('--output', help='gif output path', default='.')
parser.add_argument('--name',
                    type=str,
                    help='gif name',
                    default='generated_gif')

args = parser.parse_args()

if args.input:
    imgs_dir = args.input
else:
    raise Exception('missing --input arg, see "gif_maker.py -h" for help')

file_names = glob.glob(f'{imgs_dir}*.*')
file_names.sort()

images = []
with imageio.get_writer(f'{args.output}/{args.name}.gif', mode='I') as writer:
    for filename in file_names:
        image = imageio.imread(filename)
        writer.append_data(image)

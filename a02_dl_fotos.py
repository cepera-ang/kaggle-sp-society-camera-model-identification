# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import jpeg4py as jpeg
from PIL import Image
import struct
import imghdr
import exifread
import skimage.io
import imageio
import pyvips


def get_photos(file_with_urls, folder_to_store, prefix):
    import urllib.request
    f = open(file_with_urls)
    if not os.path.isdir(folder_to_store):
        os.mkdir(folder_to_store)

    lines = f.readlines()
    for l in lines:
        l = l.strip()
        filename = prefix + '_' + os.path.basename(l) + '.jpg'
        store_path = folder_to_store + filename
        print('URL: {}'.format(l))
        print('Store: {}'.format(store_path))
        if os.path.isfile(store_path):
            print('File already exists. Skip!')
            continue
        try:
            response = urllib.request.urlopen(l)
            html = response.read()
            out = open(store_path, 'wb')
            out.write(html)
            out.close()
        except Exception as ex:
            print('Fail: {}'.format(ex))
        time.sleep(1)


if __name__ == '__main__':
    for l in ['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'Motorola-Droid-Maxx',
              'Motorola-Nexus-6', 'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']:
        get_photos(OUTPUT_PATH + 'additional_images/yaphoto/{}.txt'.format(l), INPUT_PATH + 'raw/yaphoto/{}/'.format(l), prefix=l)

    for l in ['HTC-1-M7', 'iPhone-4s', 'iPhone-6', 'Motorola-Droid-Maxx',
              'Motorola-Nexus-6', 'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7']:
        get_photos(OUTPUT_PATH + 'additional_images/flickr3/{}/correct_shapes.txt'.format(l), INPUT_PATH + 'raw/flickr3/{}/'.format(l),
                   prefix=l)

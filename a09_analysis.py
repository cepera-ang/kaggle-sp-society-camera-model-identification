# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import jpeg4py as jpeg
from PIL import Image
import struct
import imghdr
import exifread


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        what = imghdr.what(None, head)
        if what == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif what == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif what == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height


def check_train_resolutions():
    files = glob.glob(INPUT_PATH + 'train/**/*.jpg', recursive=True)
    camera_sizes = dict()
    for f in files:
        dir = os.path.basename(os.path.dirname(f))
        sz = get_image_size(f)
        if dir not in camera_sizes:
            camera_sizes[dir] = dict()
        if sz in camera_sizes[dir]:
            camera_sizes[dir][sz] += 1
        else:
            camera_sizes[dir][sz] = 1

    for el in sorted(camera_sizes.keys()):
        print('Camera: {} Resolutions in train: {}'.format(el, camera_sizes[el]))


def check_external_resolutions():
    files = glob.glob(INPUT_PATH + 'external/**/*.jpg', recursive=True)
    camera_sizes = dict()
    for f in files:
        dir = os.path.basename(os.path.dirname(f))
        sz = get_image_size(f)
        if dir not in camera_sizes:
            camera_sizes[dir] = dict()
        if sz in camera_sizes[dir]:
            camera_sizes[dir][sz] += 1
        else:
            camera_sizes[dir][sz] = 1

    for el in sorted(camera_sizes.keys()):
        print('Camera: {} Resolutions in external: {}'.format(el, camera_sizes[el]))


def test_different_jpeg_readers():
    import numpy as np
    from PIL import Image
    import jpeg4py as jpeg
    import cv2

    a = jpeg.JPEG('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg').decode()
    print(a.shape)
    print(a)
    print('\n\n\n\n')
    show_resized_image(a)

    b = cv2.imread('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    b = np.transpose(b, (1, 0, 2))
    b = np.flip(b, axis=0)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    print(b.shape)
    print(b)
    print('\n\n\n\n')
    show_resized_image(b)

    c = Image.open('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    c = np.array(c)
    print(c.shape)
    print(c)
    show_resized_image(c)

    print('Max diff 1: {}'.format(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()))
    print('Max diff 2: {}'.format(np.abs(a.astype(np.int32) - c.astype(np.int32)).max()))
    print('Max diff 3: {}'.format(np.abs(b.astype(np.int32) - c.astype(np.int32)).max()))
    exit()


def get_cameras_quality(type='train'):
    from PythonMagick import Image
    files = glob.glob(INPUT_PATH + type + '/**/*.jpg', recursive=True)
    camera_sizes = dict()
    for f in files:
        i = Image(f)
        print(f, i.quality(), i.size().width(), i.size().height())
        dir = os.path.basename(os.path.dirname(f))
        sz = (i.quality(), i.size().width(), i.size().height())
        if dir not in camera_sizes:
            camera_sizes[dir] = dict()
        if sz in camera_sizes[dir]:
            camera_sizes[dir][sz] += 1
        else:
            camera_sizes[dir][sz] = 1

    for el in sorted(camera_sizes.keys()):
        print('Camera: {} Quality in train: {}'.format(el, camera_sizes[el]))
    exit()


def get_software_exif(type='train'):
    files = glob.glob(INPUT_PATH + type + '/**/*.jpg', recursive=True)
    count = 0
    software = dict()
    for f in files:
        tags = exifread.process_file(open(f, 'rb'))
        try:
            soft = str(tags['Image Software'])
            print(str(soft))
            if soft in software:
                software[soft] += 1
            else:
                software[soft] = 1
            count+=1
        except:
            continue
    print('Modified found: {}'.format(count))
    for el in sorted(software.keys()):
        print('Soft: {} Count: {}'.format(el, software[el]))


if __name__ == '__main__':
    # test_1()
    # check_train_resolutions()
    # check_external_resolutions()
    # get_kfold_split(4)
    # test_different_jpeg_readers()
    # get_cameras_quality('train')
    # get_cameras_quality('external')
    # get_software_exif('external')
    t, v = get_single_split(fraction=0.9, only_train=True)
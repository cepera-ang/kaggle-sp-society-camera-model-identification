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
from io import BytesIO
import math


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


def tst_different_jpeg_readers():
    import numpy as np
    from PIL import Image
    import jpeg4py as jpeg
    import cv2

    a = jpeg.JPEG('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg').decode()
    print(a.shape)
    print(a)
    print('\n\n\n\n')
    # show_resized_image(a)

    b = cv2.imread('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    b = np.transpose(b, (1, 0, 2))
    b = np.flip(b, axis=0)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    print(b.shape)
    print(b)
    print('\n\n\n\n')
    # show_resized_image(b)

    c = Image.open('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    c = np.array(c)
    print(c.shape)
    print(c)
    # show_resized_image(c)

    d = skimage.io.imread('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    d = np.array(d)
    print(d.shape)
    print(d)
    # show_resized_image(d)

    e = imageio.imread('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg')
    e = np.transpose(e, (1, 0, 2))
    e = np.flip(e, axis=0)
    print(e.shape)
    print(e)
    # show_resized_image(e)

    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }

    f = pyvips.Image.new_from_file('../input/train/Samsung-Galaxy-Note3/(GalaxyN3)7.jpg', access='sequential')
    f = np.ndarray(buffer=f.write_to_memory(),
                       dtype=format_to_dtype[f.format],
                       shape=[f.height, f.width, f.bands])
    print(f.shape)
    print(f)
    show_resized_image(f)

    print('Max diff 1: {}'.format(np.abs(a.astype(np.int32) - b.astype(np.int32)).max()))
    print('Max diff 2: {}'.format(np.abs(a.astype(np.int32) - c.astype(np.int32)).max()))
    print('Max diff 3: {}'.format(np.abs(b.astype(np.int32) - c.astype(np.int32)).max()))
    print('Max diff 4: {}'.format(np.abs(a.astype(np.int32) - d.astype(np.int32)).max()))
    print('Max diff 5: {}'.format(np.abs(b.astype(np.int32) - d.astype(np.int32)).max()))
    print('Max diff 6: {}'.format(np.abs(b.astype(np.int32) - e.astype(np.int32)).max()))
    print('Max diff 7: {}'.format(np.abs(b.astype(np.int32) - f.astype(np.int32)).max()))
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


def check_reading_speed():
    files = glob.glob(INPUT_PATH + 'train/**/*.jpg', recursive=True)
    files = files[:300]
    print('Files: {}'.format(len(files)))

    if 0:
        start_time = time.time()
        d = []
        for f in files:
            a = jpeg.JPEG(f).decode()
            d.append(a)
        print('Time to read {} for libjpeg-turbo: {:.2f} sec'.format(len(files), time.time() - start_time))

    if 0:
        start_time = time.time()
        d = []
        for f in files:
            b = cv2.imread(f)
            b = np.transpose(b, (1, 0, 2))
            b = np.flip(b, axis=0)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            d.append(b)
        print('Time to read {} for cv2 with conversion: {:.2f} sec'.format(len(files), time.time() - start_time))

    if 0:
        start_time = time.time()
        d = []
        for f in files:
            b = cv2.imread(f)
            d.append(b)
        print('Time to read {} for cv2 no conversion: {:.2f} sec'.format(len(files), time.time() - start_time))

    if 0:
        start_time = time.time()
        d = []
        for f in files:
            c = Image.open(f)
            c = np.array(c)
            d.append(c)
        print('Time to read {} for PIL: {:.2f} sec'.format(len(files), time.time() - start_time))

    if 0:
        start_time = time.time()
        d = []
        plugin = 'matplotlib'
        for f in files:
            c = skimage.io.imread(f, plugin=plugin)
            c = np.array(c)
            d.append(c)
        print('Time to read {} for skimage.io Plugin: {}: {:.2f} sec'.format(len(files), plugin, time.time() - start_time))

    if 0:
        start_time = time.time()
        d = []
        for f in files:
            c = imageio.imread(f)
            d.append(c)
        print('Time to read {} for Imageio (no rotate): {:.2f} sec'.format(len(files), time.time() - start_time))

    if 1:
        format_to_dtype = {
            'uchar': np.uint8,
            'char': np.int8,
            'ushort': np.uint16,
            'short': np.int16,
            'uint': np.uint32,
            'int': np.int32,
            'float': np.float32,
            'double': np.float64,
            'complex': np.complex64,
            'dpcomplex': np.complex128,
        }

        start_time = time.time()
        d = []
        for f in files:
            c = pyvips.Image.new_from_file(f, access='sequential')
            c = np.ndarray(buffer=c.write_to_memory(),
                           dtype=format_to_dtype[c.format],
                           shape=[c.height, c.width, c.bands])
            d.append(c)
        print('Time to read {} for PyVips: {:.2f} sec'.format(len(files), time.time() - start_time))


def check_multithread_jpeg_read():
    return


def improve_subm_v1(subm_path, out_path):
    df = pd.read_csv(subm_path)
    answ = np.argmax(df[CLASSES].values, axis=1)
    camera = np.array(CLASSES)[answ]
    df['camera'] = camera
    print(df['camera'])

    for c in CLASSES:
        print('{}: {}'.format(c, len(df[df['camera'] == c])))

    checker = dict()
    for c in CLASSES:
        checker[c] = [0, 0]

    manip = []
    for index, row in df.iterrows():
        if '_manip' in row['fname']:
            checker[row['camera']][0] += 1
            manip.append(1)
        else:
            checker[row['camera']][1] += 1
            manip.append(0)
    df['manip'] = manip

    manip_counts = []
    raw_counts = []
    for c in CLASSES:
        print('{}: {}'.format(c, checker[c]))
        manip_counts.append((checker[c][0], c))
        raw_counts.append((checker[c][1], c))

    manip_counts.sort(key=lambda tup: tup[0], reverse=True)
    raw_counts.sort(key=lambda tup: tup[0], reverse=True)
    print(manip_counts)
    print(raw_counts)

    total_iters = 0
    while 1:
        total_iters += 1

        checker = dict()
        for c in CLASSES:
            checker[c] = [0, 0]

        manip = []
        for index, row in df.iterrows():
            if '_manip' in row['fname']:
                checker[row['camera']][0] += 1
                manip.append(1)
            else:
                checker[row['camera']][1] += 1
                manip.append(0)
        df['manip'] = manip

        manip_counts = []
        raw_counts = []
        exit_flag = 1
        for c in CLASSES:
            print('{}: {}'.format(c, checker[c]))
            manip_counts.append((checker[c][0], c))
            raw_counts.append((checker[c][1], c))
            if checker[c][0] != 132 or checker[c][1] != 132:
                exit_flag = 0

        if exit_flag == 1 or total_iters > 100:
            break

        manip_counts.sort(key=lambda tup: tup[0], reverse=True)
        raw_counts.sort(key=lambda tup: tup[0], reverse=True)

        # only manip data
        manip = df[df['manip'] == 1]
        for count, c in manip_counts:
            class_part = manip[manip['camera'] == c].copy()
            class_part = class_part.sort_values(c, ascending=False)
            for i in range(132, len(class_part)):
                df.loc[class_part.index.values[i], c] = 0.0
            break

        # only raw data
        manip = df[df['manip'] == 0]
        for count, c in manip_counts:
            class_part = manip[manip['camera'] == c].copy()
            class_part = class_part.sort_values(c, ascending=False)
            for i in range(132, len(class_part)):
                df.loc[class_part.index.values[i], c] = 0.0
            break

        answ = np.argmax(df[CLASSES].values, axis=1)
        camera = np.array(CLASSES)[answ]
        df['camera'] = camera

    checker = dict()
    for c in CLASSES:
        checker[c] = [0, 0]

    for index, row in df.iterrows():
        if '_manip' in row['fname']:
            checker[row['camera']][0] += 1
        else:
            checker[row['camera']][1] += 1

    for c in CLASSES:
        print('{}: {}'.format(c, checker[c]))

    df[['fname', 'camera']].to_csv(out_path, index=False)


def check_subm_distribution(subm_path):
    df = pd.read_csv(subm_path)
    checker = dict()
    for c in CLASSES:
        checker[c] = [0, 0]

    manip = []
    for index, row in df.iterrows():
        if '_manip' in row['fname']:
            checker[row['camera']][0] += 1
            manip.append(1)
        else:
            checker[row['camera']][1] += 1
            manip.append(0)
    df['manip'] = manip

    for c in CLASSES:
        print('{}: {}'.format(c, checker[c]))


def check_subm_diff(s1p, s2p):
    df1 = pd.read_csv(s1p)
    df2 = pd.read_csv(s2p)
    df1.sort_values('fname', inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.sort_values('fname', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    dff = len(df1[df1['camera'] != df2['camera']])
    total = len(df1)
    perc = 100 * dff / total
    print('Difference in {} pos from {}. Percent: {:.2f}%'.format(dff, total, perc))


def check_image_manipulation():
    import imageio
    import PythonMagick

    test = glob.glob(INPUT_PATH + 'test/*_unalt.tif')
    img = pyvips.Image.new_from_file(test[0], access='sequential')
    img = np.ndarray(buffer=img.write_to_memory(),
                     dtype=np.uint8,
                     shape=[img.height, img.width, img.bands])

    quality = 70
    out = BytesIO()
    im = Image.fromarray(img)
    im.save(out, format='jpeg', quality=quality)
    jpeg70_1 = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()

    _, out = cv2.imencode('.jpg', img.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    jpeg70_2 = cv2.imdecode(out, 1)
    print(jpeg70_2.shape)

    # Same as PIL
    if 0:
        out = BytesIO()
        imageio.imwrite(out, img, format='jpeg', quality=quality)
        jpeg70_3 = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()

    if 0:
        out = BytesIO()
        i = PythonMagick.Image(test[0])
        i.quality(quality)
        i.write(out)
        jpeg70_3 = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()

    out = BytesIO()
    im = Image.fromarray(img)
    im.save(out, format='png')
    i = pyvips.Image.new_from_buffer(out, "")
    i = i.quality(70)
    jpeg70_3 = np.ndarray(buffer=i.write_to_memory(),
                     dtype=np.uint8,
                     shape=[i.height, i.width, i.bands])

    show_image(img)
    show_image(jpeg70_1)
    show_image(jpeg70_2)
    show_image(jpeg70_3)
    diff = (np.abs(img.astype(np.int32) - jpeg70_1.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff/(img.shape[0]*img.shape[1]*img.shape[2])))
    diff = (np.abs(img.astype(np.int32) - jpeg70_2.astype(np.int32))).sum()
    print('Pixel diff 2: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))
    diff = (np.abs(img.astype(np.int32) - jpeg70_3.astype(np.int32))).sum()
    print('Pixel diff 2: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))


def check_subm_diff_simple(s1p, s2p):
    df1 = pd.read_csv(s1p)
    df2 = pd.read_csv(s2p)
    df1.sort_values('fname', inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.sort_values('fname', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    dff = len(df1[df1['camera'] != df2['camera']])
    total = len(df1)
    perc = 100 * dff / total
    return dff, perc


def check_subm_diff_table():
    best = SUBM_PATH + 'equal_all_fix_0.983.csv'
    subms = glob.glob(SUBM_PATH + 'subm_with_score/*.csv')
    for s in subms:
        dff, perc = check_subm_diff_simple(best, s)
        print("Score on LB {} Diff with best {} ({:.2f} %)".format(os.path.basename(s).split('_')[0], dff, perc))


def check_gamma_change():
    test = glob.glob(INPUT_PATH + 'test/*_unalt.tif')
    img = pyvips.Image.new_from_file(test[0], access='sequential')
    img = np.ndarray(buffer=img.write_to_memory(),
                     dtype=np.uint8,
                     shape=[img.height, img.width, img.bands])

    gamma = 1.2
    img1 = skimage.exposure.adjust_gamma(img, gamma)
    img2 = np.uint8(cv2.pow(img / 255., gamma) * 255.)
    show_image(img)
    show_image(img1)
    show_image(img2)
    diff = (np.abs(img1.astype(np.int32) - img2.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))


def get_crop(img, crop_size, random_crop=False):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop = crop_size // 2
    pad_x = max(0, crop_size - img.shape[1])
    pad_y = max(0, crop_size - img.shape[0])
    if (pad_x > 0) or (pad_y > 0):
        img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2), (0,0)), mode='wrap')
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    if random_crop:
        freedom_x, freedom_y = img.shape[1] - crop_size, img.shape[0] - crop_size
        if freedom_x > 0:
            center_x += np.random.randint(math.ceil(-freedom_x/2), freedom_x - math.floor(freedom_x/2) )
        if freedom_y > 0:
            center_y += np.random.randint(math.ceil(-freedom_y/2), freedom_y - math.floor(freedom_y/2) )

    return img[center_y - half_crop : center_y + crop_size - half_crop, center_x - half_crop : center_x + crop_size - half_crop]


def check_bicubic_change():
    import scipy.misc
    import skimage.transform

    train = glob.glob(INPUT_PATH + 'train/*/*.jpg')
    img = pyvips.Image.new_from_file(train[0], access='sequential')
    img = np.ndarray(buffer=img.write_to_memory(),
                     dtype=np.uint8,
                     shape=[img.height, img.width, img.bands])

    bicubic = 2.0
    img = get_crop(img, 512 * 2, random_crop=False)
    img0 = get_crop(img, 512, random_crop=False)

    img1 = cv2.resize(img, (0, 0), fx=bicubic, fy=bicubic, interpolation=cv2.INTER_CUBIC)
    img1 = get_crop(img1, 512, random_crop=False)

    img2 = scipy.misc.imresize(img, bicubic, interp='bicubic')
    img2 = get_crop(img2, 512, random_crop=False)

    img3 = (255. * skimage.transform.rescale(img, bicubic, order=3, mode='constant')).astype(np.uint8)
    img3 = get_crop(img3, 512, random_crop=False)
    print(img3.shape, img3.dtype, img3.min(), img3.max())

    show_image(img1)
    show_image(img2)
    show_image(img3)
    diff = (np.abs(img0.astype(np.int32) - img1.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))
    diff = (np.abs(img0.astype(np.int32) - img2.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))
    diff = (np.abs(img1.astype(np.int32) - img2.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))
    diff = (np.abs(img1.astype(np.int32) - img3.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))
    diff = (np.abs(img2.astype(np.int32) - img3.astype(np.int32))).sum()
    print('Pixel diff 1: {} avg: {}'.format(diff, diff / (img.shape[0] * img.shape[1] * img.shape[2])))


if __name__ == '__main__':
    # test_1()
    # check_train_resolutions()
    # check_external_resolutions()
    # get_kfold_split(4)
    # test_different_jpeg_readers()
    # get_cameras_quality('train')
    # get_cameras_quality('external')
    # get_software_exif('external')
    # t, v = get_single_split(fraction=0.9, only_train=True)
    # tst_different_jpeg_readers()
    # check_reading_speed()
    # improve_subm_v1(SUBM_PATH + '3_sq_mean_raw.csv', SUBM_PATH + '3_sq_mean_fixed.csv')
    check_subm_distribution(SUBM_PATH + 'submission_resnet50_antorsaegen_119_val_0.9815068493150685_tta_arithmetic.csv')
    check_subm_diff(SUBM_PATH + '0.985_equal_2_power_mean_hun.csv', SUBM_PATH + 'submission_resnet50_antorsaegen_119_val_0.9815068493150685_tta_arithmetic.csv')
    # get_single_split_final(OUTPUT_PATH + 'common_image_info_additional.csv', OUTPUT_PATH + 'validation_files.pklz')
    # check_image_manipulation()
    # check_subm_diff_table()
    # check_gamma_change()
    # check_bicubic_change()

'''
Time to read 300 for libjpeg-turbo: 9.62 sec
Time to read 300 for cv2 with conversion: 41.68 sec
Time to read 300 for cv2 no conversion: 24.69 sec
Time to read 300 for PIL: 27.91 sec
Time to read 300 for skimage.io: 30.06 sec
Time to read 300 for skimage.io Plugin: matplotlib: 30.43 sec
Time to read 300 for skimage.io Plugin: freeimage: 30.80 sec
Time to read 300 for Imageio (no rotate): 26.10 sec

Time to read 300 for libjpeg-turbo: 9.86 sec
Time to read 300 for cv2 with conversion: 36.10 sec
Time to read 300 for cv2 no conversion: 21.70 sec
Time to read 300 for PIL: 25.17 sec
Time to read 300 for PIL-simd: 25.81 sec
Time to read 300 for skimage.io Plugin: freeimage: 27.80 sec
Time to read 300 for Imageio (no rotate): 24.55 sec
Time to read 300 for PyVips: 12.61 sec

0.978 subm distribution
HTC-1-M7: [133, 131]
iPhone-6: [132, 132]
Motorola-Droid-Maxx: [131, 133]
Motorola-X: [133, 131]
Samsung-Galaxy-S4: [133, 131]
iPhone-4s: [132, 132]
LG-Nexus-5x: [126, 138]
Motorola-Nexus-6: [130, 134]
Samsung-Galaxy-Note3: [135, 129]
Sony-NEX-7: [135, 129]

'''
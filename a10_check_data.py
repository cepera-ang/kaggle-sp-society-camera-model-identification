# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import hashlib
import exifread
from a00_common_functions import *


def md5_from_file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(path):
    files = glob.glob(path + '**/*.jpg', recursive=True)
    out = open(OUTPUT_PATH + 'files_hash_stat.csv', 'w')
    out.write('path,md5\n')
    print('Files found: {}'.format(len(files)))
    all_hashes = dict()
    for f in files:
        # print('Go for {}'.format(f))
        hsh = md5_from_file(f)
        out.write(f + ',' + hsh + '\n')
        if hsh in all_hashes:
            all_hashes[hsh].append(f)
        else:
            all_hashes[hsh] = [f]
    out.close()

    duplicate_count = 0
    for el in all_hashes:
        if len(all_hashes[el]) > 1:
            print('Duplicate found. Count: {}. List below'.format(len(all_hashes[el])))
            for item in all_hashes[el]:
                print(item)
            duplicate_count += (len(all_hashes[el]) - 1)
    print('Total duplicates found: {}'.format(duplicate_count))


def get_train_exif_dict(path):
    files = glob.glob(path + '**/*.jpg', recursive=True)
    out = open(OUTPUT_PATH + 'train_exif_stat.csv', 'w')
    out.write('path,model\n')
    exif_dict = dict()
    print('Files found: {}'.format(len(files)))
    for f in files:
        tags = exifread.process_file(open(f, 'rb'))
        dir = os.path.basename(os.path.dirname(f))
        model = tags['Image Model']
        if dir not in exif_dict:
            exif_dict[dir] = []
        if str(model) not in exif_dict[dir]:
            print('Append {}'.format(model))
            exif_dict[dir].append(str(model))
        out.write(f + ',' + str(model) + '\n')
        # print(model)
    for el in sorted(exif_dict.keys()):
        print('Folder {}: Available models: {}'.format(el, exif_dict[el]))
    out.close()
    return exif_dict


def get_external_exif_dict(path):
    files = glob.glob(path + '**/*.jpg', recursive=True)
    out = open(OUTPUT_PATH + 'exif_stat_external.csv', 'w')
    out.write('path,model\n')
    exif_dict = dict()
    print('Files found: {}'.format(len(files)))
    for f in files:
        tags = exifread.process_file(open(f, 'rb'))
        dir = os.path.basename(os.path.dirname(f))
        try:
            model = tags['Image Model']
        except:
            out.write(f + '\n')
            out.write(str(tags) + '\n')
            out.write('\n\n\n\n')
            model = ''
        if dir not in exif_dict:
            exif_dict[dir] = dict()
        if str(model) not in exif_dict[dir]:
            print('Append {}'.format(model))
            exif_dict[dir][str(model)] = 1
        else:
            exif_dict[dir][str(model)] += 1
        # out.write(f + ',' + str(model) + '\n')
        # print(model)
    for el in sorted(exif_dict.keys()):
        print('Folder {}: Available models: {}'.format(el, exif_dict[el]))
    out.close()
    return exif_dict


def prepare_external_dataset(raw_path, output_path):
    exif_dict = {
        'HTC One': 'HTC-1-M7',
        'HTC6500LVW': 'HTC-1-M7',
        'HTCONE': 'HTC-1-M7',

        'Nexus 5X': 'LG-Nexus-5x',

        'XT1080': 'Motorola-Droid-Maxx',
        'XT1060': 'Motorola-Droid-Maxx',

        'Nexus 6': 'Motorola-Nexus-6',

        'XT1096': 'Motorola-X',
        'XT1092': 'Motorola-X',
        'XT1095': 'Motorola-X',
        'XT1097': 'Motorola-X',
        'XT1093': 'Motorola-X',

        'SAMSUNG-SM-N900A': 'Samsung-Galaxy-Note3',
        'SM-N9005': 'Samsung-Galaxy-Note3',
        'SM-N900P': 'Samsung-Galaxy-Note3',

        'SCH-I545': 'Samsung-Galaxy-S4',
        'GT-I9505': 'Samsung-Galaxy-S4',
        'SPH-L720': 'Samsung-Galaxy-S4',

        'NEX-7': 'Sony-NEX-7',

        'iPhone 4S': 'iPhone-4s',

        'iPhone 6': 'iPhone-6',
        'iPhone 6 Plus': 'iPhone-6',
    }

    hash_checker = dict()
    files = glob.glob(raw_path + '**/*.jpg', recursive=True)
    if os.path.isdir(output_path):
        print('Folder "{}" already exists! You must delete it before proceed!'.format(output_path))
        exit()
    os.mkdir(output_path)
    print('Files found: {}'.format(len(files)))
    for f in files:
        tags = exifread.process_file(open(f, 'rb'))
        try:
            model = str(tags['Image Model'])
        except:
            print('Broken Image Model EXIF: {}'.format(f))
            continue
        if model not in exif_dict:
            print('Skip EXIF {}'.format(model))
            continue

        # Check unique hash
        hsh = md5_from_file(f)
        if hsh in hash_checker:
            print('Hash {} for file {} alread exists. Skip file!'.format(hsh, f))
            continue
        hash_checker[hsh] = 1

        out_folder = output_path + exif_dict[model]
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        shutil.copy2(f, out_folder)

    copied_files = glob.glob(output_path + '**/*.jpg', recursive=True)
    print('Files in external folder: {}'.format(len(copied_files)))
    return exif_dict


if __name__ == '__main__':
    # find_duplicates(INPUT_PATH + 'external/')
    # find_duplicates(INPUT_PATH)
    # get_train_exif_dict(INPUT_PATH + 'train/')
    # get_external_exif_dict(INPUT_PATH + 'raw/')
    prepare_external_dataset(INPUT_PATH + 'raw/', INPUT_PATH + 'external/')


'''
Folder HTC-1-M7: Available models: ['HTC One', 'HTC6500LVW']
Folder LG-Nexus-5x: Available models: ['Nexus 5X']
Folder Motorola-Droid-Maxx: Available models: ['XT1080']
Folder Motorola-Nexus-6: Available models: ['Nexus 6']
Folder Motorola-X: Available models: ['XT1096']
Folder Samsung-Galaxy-Note3: Available models: ['SAMSUNG-SM-N900A']
Folder Samsung-Galaxy-S4: Available models: ['SCH-I545']
Folder Sony-NEX-7: Available models: ['NEX-7']
Folder iPhone-4s: Available models: ['iPhone 4S']
Folder iPhone-6: Available models: ['iPhone 6']

Folder HTC-1-M7: Available models: {'HTC One': 1991}
Folder LG-Nexus-5x: Available models: {'Nexus 5X': 1726}
Folder Motorola-Droid-Maxx: Available models: {'XT1060': 2043}
Folder Motorola-Nexus-6: Available models: {'Nexus 6': 2037}
Folder Motorola-X: Available models: {'': 1, 'XT1092': 83, 'XT1060': 1, 'XT1095': 135, 'XT1097': 132, 'XT1093': 30, 'XT1096': 432}
Folder Samsung-Galaxy-Note3: Available models: {'SM-N9005': 1696}
Folder Samsung-Galaxy-S4: Available models: {'GT-I9505': 1777}
Folder Sony-NEX-7: Available models: {'NEX-7': 2033}
Folder htc_m7: Available models: {'': 1, 'HTC One': 798, 'HTCONE': 45}
Folder iPhone-4s: Available models: {'iPhone 4S': 1950}
Folder iPhone-6: Available models: {'iPhone 6': 2039}
Folder iphone_4s: Available models: {'': 38, 'iPhone 4S': 1272}
Folder iphone_6: Available models: {'iPhone 6': 1281, '': 2, 'iPhone 6 Plus': 30}
Folder moto_maxx: Available models: {'XT1080': 705}
Folder moto_x: Available models: {'XT1092': 30, 'XT1060': 1622}
Folder nexus_5x: Available models: {'': 92, 'NIKON D90': 1, 'FRD-L02': 1, 'Vignette for Android': 14, 'Nexus 5X': 595, 'Canon EOS 5D Mark III': 3, 'E-M10           ': 1}
Folder nexus_6: Available models: {'Nexus 6': 1228}
Folder samsung_note3: Available models: {'': 2, 'SM-N900P': 18, 'SM-N9005': 1450}
Folder samsung_s4: Available models: {'': 2, 'GT-I9505': 1296, 'SPH-L720': 44}
Folder sony_nex7: Available models: {'': 2, 'NEX-7': 1345}
'''
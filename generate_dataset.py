from os import listdir, makedirs
from os.path import join, isdir

import imageio
import cv2

DATA_DIR = 'path/to/daisdataset'
N_MACHINES = 38
N_TRAIN_MACHINES = 32
PATTERNS = {
    '0_0_0': 'no corrosion',
    '1_0_0': 'soft corrosion',
    '1_1_0': 'medium corossion',
    '1_1_1': 'hard corossion',
    '2_0_0': 'soft damage'
}
OK_CLASS = ['0_0_0']


# dataset generation functions. Reorganize raw data to format used by training scripts


def generate_dataset(orig_size=True):
    inspections = [inspection for inspection in listdir(DATA_DIR) if isdir(join(DATA_DIR, inspection))]
    files = []
    for inspection in inspections:
        for file in listdir(join(DATA_DIR, inspection)):
            if file.lower().endswith('bmp'):
                files.append(join(DATA_DIR, inspection, file))

    img_num = 0
    for i in range(1, N_MACHINES):
        print(i)
        object_files = [file for file in files if file.split('/')[2].split('_')[0] == str(i + 1)]

        for file in object_files:
            filename = file.split('/')[-1]
            object_class = file.split('.')[0][-5:]

            if object_class in PATTERNS.keys():
                image = imageio.imread(file)
                if orig_size:
                    resized_image_nearest = image
                else:
                    resized_image_nearest = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
                mode = 'train' if i <= N_TRAIN_MACHINES else 'test'
                class_name = 'ok' if object_class in OK_CLASS else 'corrosion'
                dataset_name = 'dataset_orig' if orig_size else 'dataset_320'

                save_path = join('data', dataset_name, mode, class_name, f'{img_num}_{filename}')
                img_num += 1
                imageio.imsave(save_path, resized_image_nearest)
                imageio.imread(save_path)
            else:
                print(file)


def generate_dataset_4class(oryg_size=True):
    inspections = [inspection for inspection in listdir(DATA_DIR) if isdir(join(DATA_DIR, inspection))]
    files = []
    for inspection in inspections:
        for file in listdir(join(DATA_DIR, inspection)):
            if file.lower().endswith('bmp'):
                files.append(join(DATA_DIR, inspection, file))

    img_num = 0
    for i in range(1, N_MACHINES):
        print(i)
        object_files = [file for file in files if file.split('/')[2].split('_')[0] == str(i + 1)]

        for file in object_files:
            filename = file.split('/')[-1]
            object_class = file.split('.')[0][-5:]

            if object_class in PATTERNS.keys():
                image = imageio.imread(file)
                if oryg_size:
                    resized_image_nearest = image
                else:
                    resized_image_nearest = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
                mode = 'train' if i <= N_TRAIN_MACHINES else 'test'
                # class_name = 'ok' if object_class in OK_CLASS else 'corrosion'
                if object_class == '0_0_0':
                    class_name = '0'
                elif object_class == '1_0_0':
                    class_name = '1'
                elif object_class == '1_1_0':
                    class_name = '2'
                elif object_class == '1_1_1':
                    class_name = '3'
                else:
                    continue

                dataset_name = 'dataset4_orig' if oryg_size else 'dataset4_320'

                save_path = join('data', dataset_name, mode, class_name, f'{img_num}_{filename}')
                img_num += 1
                imageio.imsave(save_path, resized_image_nearest)
                imageio.imread(save_path)
            else:
                print(file)


def generate_dataset_split_by_machine(oryg_size=True, train_part=0.8):
    dataset_name = 'dataset_orig_split' if oryg_size else 'dataset_320_split'

    inspections = [inspection for inspection in listdir(DATA_DIR) if isdir(join(DATA_DIR, inspection))]
    files = []
    for inspection in inspections:
        for file in listdir(join(DATA_DIR, inspection)):
            if file.lower().endswith('bmp'):
                files.append(join(DATA_DIR, inspection, file))

    img_num = 0
    for i in range(1, N_MACHINES):
        makedirs(join('data', dataset_name, str(i), 'train', 'ok'), exist_ok=True)
        makedirs(join('data', dataset_name, str(i), 'test', 'ok'), exist_ok=True)
        makedirs(join('data', dataset_name, str(i), 'train', 'corrosion'), exist_ok=True)
        makedirs(join('data', dataset_name, str(i), 'test', 'corrosion'), exist_ok=True)

        object_files = [file for file in files if file.split('/')[2].split('_')[0] == str(i)]

        good_image_n = len([_ for _ in object_files if _.split('.')[0][-5:] in PATTERNS.keys()])
        good_image = 0
        for file in object_files:
            filename = file.split('/')[-1]
            dirname = file.split('/')[-2]
            object_class = file.split('.')[0][-5:]

            if object_class in PATTERNS.keys():
                good_image += 1
                image = imageio.imread(file)
                if oryg_size:
                    resized_image_nearest = image
                else:
                    resized_image_nearest = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
                mode = 'train' if good_image / good_image_n <= train_part else 'test'
                class_name = 'ok' if object_class in OK_CLASS else 'corrosion'

                save_path = join('data', dataset_name, str(i), mode, class_name, f'{dirname}_{img_num}_{filename}')
                img_num += 1
                imageio.imsave(save_path, resized_image_nearest)
            else:
                print(file)


if __name__ == '__main__':
    generate_dataset(oryg_size=True)
    generate_dataset(oryg_size=False)
    generate_dataset_4class(oryg_size=True)
    generate_dataset_split_by_machine(oryg_size=False)

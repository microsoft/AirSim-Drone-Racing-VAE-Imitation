import os
import csv
import argparse

import pandas as pd


TIME_COLUMN = 'TimeStamp'
INTERPOLABLE_COLUMNS = ['vx', 'vy', 'vz', 'vyaw']
IMAGE_COLUMNS = [TIME_COLUMN, 'ImageFile']
RESULT_COLUMNS = INTERPOLABLE_COLUMNS


def get_abspath(filename):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), './{}'.format(filename)))


def create_image_path(image_file_name, image_folder_path):
    return os.path.abspath(os.path.join(image_folder_path, image_file_name))


def create_suffixed_file(file_path, suffix):
    _path, _format = os.path.splitext(file_path)
    return '{}_{}{}'.format(_path, suffix, _format)


def interpolate(v0, v1, t):
    return round((1 - t) * v0 + t * v1, 8)


def normalize(v0, v1, x):
    return (x - v0) / (v1 - v0)


def interpolate_record(record1, record2, image_record):
    """
    Returns result record with interpolated values
    """
    interpolated_record = {}

    for col in INTERPOLABLE_COLUMNS:
        t = normalize(
            record1[TIME_COLUMN], record2[TIME_COLUMN], image_record[TIME_COLUMN])
        interpolated_record[col] = interpolate(
            record1[col], record2[col], t)

    return interpolated_record


def find_closest_rows(value, iterator):
    v1, v2 = None, None
    for current in iterator:
        curr_value = current[1]
        if curr_value[TIME_COLUMN] <= value:
            v1 = curr_value
        elif v1 is not None and curr_value[TIME_COLUMN] >= value:
            v2 = curr_value
            break
        elif v1 is None and curr_value[TIME_COLUMN] >= value:
            break
    return v1, v2


def split_test_training_data(file_paths, lines_number, test_split=0.2):
    test_number = int(lines_number * test_split)
    for file_path in file_paths:
        f = open(file_path, 'r')
        f_test = open(create_suffixed_file(file_path, 'test'), 'w')
        f_train = open(create_suffixed_file(file_path, 'train'), 'w')

        i = 0
        for line in f.readlines():
            if i <= test_number:
                f_test.writelines(line)
            else:
                f_train.writelines(line)
            i += 1

        f.close()
        f_train.close()
        f_test.close()
        os.remove(file_path)


def process(
    velocities,
    images,
    result_velocities_file_path,
    result_images_file_path,
    images_folder_path
):
    """
    Process velocities and images frames.
    For each row in images:
        1) Match 2 closest by timestamp velocities rows to the image record.
        2) Calculate normalized parameter t: image_time - vt1 / vt2 - vt1.
           vt1, vt2: velocity records timestamps
        3) Interpolate velocities values using t.
        4) Create new row using image timestamp, image and interpolated values.
    """
    velocity_iterator = velocities.iterrows()
    f_velocities = open(result_velocities_file_path, 'w+')
    f_images = open(result_images_file_path, 'w+')
    writer_v = csv.DictWriter(f_velocities, RESULT_COLUMNS, delimiter=',')
    writer_i = csv.DictWriter(f_images, ['ImageFile'], delimiter=',')
    row_counter, missed = 0, 0

    for _, image_row in images.iterrows():
        v1, v2 = find_closest_rows(image_row[TIME_COLUMN], velocity_iterator)
        if v1 is None or v2 is None:
            continue
        interpolated = interpolate_record(v1, v2, image_row)
        row_counter += 1

        image_path = create_image_path(
            image_row['ImageFile'],
            images_folder_path
        )

        if not os.path.isfile(image_path):
            missed += 1
            continue

        writer_v.writerow(interpolated)
        writer_i.writerow({
            'ImageFile': image_path
        })

    print('--------------------------------')
    print('Missed files: {}'.format(missed))
    f_velocities.close()
    f_images.close()

    split_test_training_data(
        [result_velocities_file_path, result_images_file_path], row_counter)


def run(
    velocities_file_path,
    images_file_path,
    result_velocities_file_path,
    result_images_file_path,
    images_folder_path,
):
    velocities = pd.read_csv(velocities_file_path, delimiter=', ')
    images = pd.read_csv(
        images_file_path, delimiter=', ')

    process(
        velocities,
        images,
        result_velocities_file_path,
        result_images_file_path,
        images_folder_path
    )
    print('------------------------------------')
    print('Successfully created the results!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("velocity", help="Path to the velocities file")
    parser.add_argument("images", help="Path to the images file")
    parser.add_argument(
        "result_velocities", help="Path to the result velocities file")
    parser.add_argument("result_images", help="Path to the result images file")
    parser.add_argument("images_folder", help="Path to the images folder")
    args = parser.parse_args()
    run(
        args.velocity,
        args.images,
        args.result_velocities,
        args.result_images,
        args.images_folder
    )

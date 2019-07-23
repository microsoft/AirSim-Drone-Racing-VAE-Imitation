import os
import csv
import argparse

import pandas as pd


TIME_COLUMN = 'TimeStamp'
INTERPOLABLE_COLUMNS = ['vx', 'vy', 'vz', 'vyaw']
IMAGE_COLUMNS = [TIME_COLUMN, 'ImageFile']
RESULT_COLUMNS = [TIME_COLUMN] + INTERPOLABLE_COLUMNS + ['ImageFile']


def get_abspath(filename):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), './{}'.format(filename)))


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

    for col in IMAGE_COLUMNS:
        interpolated_record[col] = image_record[col]

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


def process(velocities, images, result_file_path):
    """
    Process velocities and images frames.
    For each row in images:
        1) Match 2 closest by timestamp velocities rows to the image record.
        2) Calculate normalized parameter t: image_time - vt1 / vt2 - vt1.
           vt1, vt2: velocity records timestamps
        3) Interpolate velocities values using t.
        4) Create new row using image timestamp, image and interpolated values.
    :param velocities: pd.DataFrame
    :param images: pd.DataFrame
    """
    velocity_iterator = velocities.iterrows()
    f = open(result_file_path, 'w+')
    writer = csv.DictWriter(f, RESULT_COLUMNS, delimiter=',')
    writer.writeheader()

    for _, image_row in images.iterrows():
        v1, v2 = find_closest_rows(image_row[TIME_COLUMN], velocity_iterator)
        if v1 is None or v2 is None:
            continue
        writer.writerow(interpolate_record(v1, v2, image_row))

    f.close()


def run(velocities_file_path, images_file_path, result_file_path):
    velocities = pd.read_csv(velocities_file_path, delimiter=', ')
    images = pd.read_csv(
        images_file_path, delimiter=', ')

    process(velocities, images, result_file_path)
    print('------------------------------------')
    print('Successfully written the result to: ', result_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("velocity", help="Path to the velocities file")
    parser.add_argument("images", help="Path to the images file")
    parser.add_argument("result", help="Path to the result file")
    parser.add_argument(
        "-rel",
        "--relative",
        help="Enable relative paths to load "
             "files from the folder where this script is located",
        action="store_true"
    )
    args = parser.parse_args()
    if args.relative:
        run(
            get_abspath(args.velocity),
            get_abspath(args.images),
            get_abspath(args.result)
        )
    else:
        run(args.velocity, args.images, args.result)

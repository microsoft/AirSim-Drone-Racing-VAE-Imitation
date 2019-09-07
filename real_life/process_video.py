import numpy as np
import cv2
import os

input_video_path = '/home/rb/data/real_life/raw/IMG_3886.MOV'
output_path = '/home/rb/data/real_life/video_10'
img_res = 64
vis = True

# create directory for the images
path_images = os.path.join(output_path, 'images')
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(path_images)
    print('Directory {} created'.format(output_path))
else:
    print('Directory {} already exists'.format(output_path))
    exit()

cap = cv2.VideoCapture(input_video_path)

idx_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Total number of frames = {}'.format(idx_max))

idx = 0
while (idx < idx_max):
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

    # rotate to correct vertical alignment
    rows, cols, num_channels = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    # img = cv2.warpAffine(img, M, (cols, rows))

    # crop square frame of interest 1080x1080
    img = img[0:1080, 420:1500]

    # resize to desired resolution
    img = cv2.resize(img, (img_res, img_res))

    # write to png
    cv2.imwrite(os.path.join(path_images, str(idx).zfill(len(str(idx_max))) + '.png'), img)

    # visualize if we want
    if vis:
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    idx = idx + 1

cap.release()
cv2.destroyAllWindows()

print('Done processing video {}'.format(input_video_path))
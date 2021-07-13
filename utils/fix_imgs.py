import os
import cv2
import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

"""
the jpgs may be damaged, we can't feed them directly into tensorflow, this py is to fix these images
"""

img_path = '/home/zhang.xinxi/CV/data/dabiao/man_under_paikou/'
count = 0
for i, j, files in os.walk(img_path):
    for file in tqdm.tqdm(files):
        count += 1
        try:
            path = i + '/' + file
            img = tf.image.decode_image(
                open(path, 'rb').read(), channels=3)
            img = cv2.resize(img.numpy(), (480, 480))
            cv2.imwrite(i + '/' + file, img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except BaseException:
            try:
                os.remove(i + '/' + file)
                print('the file has been fixed, file=' + i + '/' + file)
            except BaseException:
                print('fixed denied, please run the following code to fix the img:')
                print('sudo rm ' + i + '/' + file)
                pass

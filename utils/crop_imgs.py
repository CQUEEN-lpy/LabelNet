"""
import
"""
import os
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from absl import app
from model.yolov3_tf2.models import (
    YoloV3
)
import tensorflow as tf
import cv2
"""
basic parameters
"""
checkpoint_table = {}
checkpoint_table['lianyiqun'] ='/home/zhang.xinxi/Python_PJ/my_YOLO_V3/checkpoints/top/yolov3_train_50.tf'
checkpoint_table['top'] = '/home/zhang.xinxi/Python_PJ/my_YOLO_V3/checkpoints/top/yolov3_train_8.tf'
checkpoint_table['under'] = '/home/zhang.xinxi/Python_PJ/my_YOLO_V3/checkpoints/under/yolov3_train_8.tf'
num_class = 2
img_size = 480

"""
runing parameters
"""
mode = 'lianyiqun'
img_path = '/home/zhang.xinxi/CV/data/dabiao/man_top_color/black_white_tiaowen'
save_path = '/home/zhang.xinxi/CV/data/dabiao/man_top_color/crop/black_white_tiaowen'
print(save_path)

def flip(i):
    if i < 0:
        return 0
    if i > 480:
        return 480
    return i

def crop(img_path =img_path, save_path=save_path, mode='lianyiqun'):
    checkpoint_path = checkpoint_table[mode]
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    
    # let memory of the GPU grow
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    
    yolo = YoloV3(classes=num_class)
    yolo.load_weights(checkpoint_path).expect_partial()
    count = 0
    for i, j,files in os.walk(img_path):
        for file in tqdm.tqdm(files):
            try:
                path = img_path + '/' + file
                img = tf.image.decode_image(
                    open(path, 'rb').read(), channels=3)
                img = cv2.resize(img.numpy(), (480, 480))
                img = tf.convert_to_tensor(img)
                img = tf.expand_dims(img, 0)
                img_2 = img/255
                boxes, scores, classes, nums = yolo(img_2)

                # crop bbox
                x1 = flip(int(480 * float(boxes[0][0][0])))
                y1 = flip(int(480 * float(boxes[0][0][1])))
                x2 = flip(int(480 * float(boxes[0][0][2])))
                y2 = flip(int(480 * float(boxes[0][0][3])))
                img = tf.reshape(img, [480, 480, 3]).numpy()
                if abs((x1-x2)*(y1-y2)) < 1000:
                    continue
                cv2.imwrite(filename=save_path+ '/' +file, img=cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
                count += 1
            except BaseException as e:
                pass
    print(count)

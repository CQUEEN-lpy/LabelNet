'''
import
'''
import os
import pandas
import tqdm
from model.EFF import *
from PIL import Image
import numpy as np

'''
basic parameter config
'''
# define the img shape that we want to feed to the model
img_width = 480
img_height = 480

"""
the following train function is for predicting the unlabeled images and generate a csv to log the information 
Parameters:
1. class_names:the class_names of the task, the index must be unanimous as training, they are usually alphabetically indexed
   (e.g. ['black', 'black_white_tiaowen', 'micai', 'other', 'white'])
2. checkpoint_path: the checkpoint path of the weight of the model
3. file_path: path that stores the unlabeled images
4. save_dir: the csv path to save the logging csv
5. stop: the max number to predict, used when testing
6. index: range from 1 to 4, to split the images to 4 parts so that we can use all 4 GPU when predicting, this parameter
   is only meaningful when the mode is not 'all'
7. batch_size: the batch_size of the data to enter the GPU when predicting
8. binary: whether the CLS task is binary 
"""

def predict(class_names, checkpoint_path, file_path, save_path, stop=3000000, index = 1, mode='all',batch_size = 128, binary=False):
    class_nums = len(class_names)
    csv_path = save_path

    """
    build the model and load it from the check point ===================================================================
    """

    model = get_EFFmodel(img_height, img_width, class_nums=class_nums, binary=binary)
    model.load_weights(checkpoint_path)

    """
    predict============================================================================================================
    """


    file_names = os.listdir(file_path)

    j = 1               # batch index
    images = []         # image column
    paths = []          # path column
    classes = []        # class column
    final = None        # the final pandas
    count = 0           # count the number of image

    # split the data in 4 parts and get the correspond part
    length = len(file_names)
    if 'all' not in mode:
        begin = int(length * ((index-1)/4))
        end = int(length * ((index)/4))
        file_names = file_names[begin:end]
        print('===========================================================================================')
        print(begin, end)
        csv_path += '_' + str(index) + '.csv'

    if not binary:
        for i in tqdm.tqdm(file_names):
            count += 1
            if count > stop:
                break
            try:
                if j > batch_size:
                    #print(count)
                    images = tf.convert_to_tensor(images, dtype='float32')
                    outputs = np.array(model(images))
                    index = np.array(tf.math.argmax(outputs, 1))
                    #print(index.size)
                    scores = []
                    for k in range(index.size):
                        scores.append(outputs[k][index[k]])
                        classes.append(class_names[index[k]])
                    index = np.expand_dims(index, 1)
                    scores = np.expand_dims(np.array(scores), 1)
                    classes = np.expand_dims(np.array(classes), 1)
                    paths = np.expand_dims(np.array(paths), 1)
                    total = np.concatenate((paths, index), 1)
                    total = np.concatenate((total, scores), 1)
                    total = np.concatenate((total, classes), 1)
                    if final is not None:
                        final = np.concatenate((final, total), 0)
                    else:
                        final = total
                    paths = []
                    images = []
                    classes = []
                    j = 1
            except BaseException as ex:
                print(ex)
                #print('111111')
                paths = []
                images = []
                classes = []
                j = 1

            try:
                img = np.array(Image.open(file_path + '/' + i))
                img = tf.image.resize(img, (img_height, img_width))

                if img.shape == (img_height, img_width, 3):
                    j += 1
                    paths.append(i)
                    images.append(img)
            except BaseException as ex:
                print(ex)

        #print(count)
        images = tf.convert_to_tensor(images, dtype='float32')
        outputs = np.array(model(images))
        index = np.array(tf.math.argmax(outputs, 1))
        scores = []
        for k in range(index.size):
            scores.append(outputs[k][index[k]])
            classes.append(class_names[index[k]])

        index = np.expand_dims(index, 1)
        scores = np.expand_dims(np.array(scores), 1)
        classes = np.expand_dims(np.array(classes), 1)
        paths = np.expand_dims(np.array(paths), 1)

        total = np.concatenate((paths, index), 1)
        total = np.concatenate((total, scores), 1)
        total = np.concatenate((total, classes), 1)
        if final is not None:
            final = np.concatenate((final, total), 0)
        else:
            final = total
    # binary predict=====================================================
    else:
        for i in tqdm.tqdm(file_names):
            if count > stop:
                break
            try:
                if j > batch_size:
                    # print(count)
                    images = tf.convert_to_tensor(images, dtype='float32')
                    outputs = np.array(model(images))
                    one = tf.ones_like(outputs)
                    zero = tf.zeros_like(outputs)
                    index = np.array(tf.where(outputs < 0.5, x=zero, y=one), dtype='int32')
                    # print(index.size)
                    scores = []
                    for k in range(index.size):
                        scores.append(outputs[k][0])
                        classes.append(class_names[index[k][0]])
                    scores = np.expand_dims(np.array(scores), 1)
                    classes = np.expand_dims(np.array(classes), 1)
                    paths = np.expand_dims(np.array(paths), 1)
                    total = np.concatenate((paths, index), 1)
                    total = np.concatenate((total, scores), 1)
                    total = np.concatenate((total, classes), 1)
                    if final is not None:
                        final = np.concatenate((final, total), 0)
                    else:
                        final = total
                    paths = []
                    images = []
                    classes = []
                    j = 1
            except BaseException as ex:
                print(ex)
                paths = []
                images = []
                classes = []
                j = 1
            try:
                img = np.array(Image.open(file_path + '/' + i))
                img = tf.image.resize(img, (img_height, img_width))
    
                if img.shape == (img_height, img_width, 3):
                    j += 1
                    paths.append(i)
                    images.append(img)
            except BaseException as ex:
                print(ex)
        try:
            if j <= batch_size:
                # print(count)
                images = tf.convert_to_tensor(images, dtype='float32')
                outputs = np.array(model(images))
                one = tf.ones_like(outputs)
                zero = tf.zeros_like(outputs)
                index = np.array(tf.where(outputs < 0.5, x=zero, y=one), dtype='int32')
                # print(index.size)
                scores = []
                for k in range(index.size):
                    scores.append(outputs[k][0])
                    classes.append(class_names[index[k][0]])
                scores = np.expand_dims(np.array(scores), 1)
                classes = np.expand_dims(np.array(classes), 1)
                paths = np.expand_dims(np.array(paths), 1)
                total = np.concatenate((paths, index), 1)
                total = np.concatenate((total, scores), 1)
                total = np.concatenate((total, classes), 1)
                if final is not None:
                    final = np.concatenate((final, total), 0)
                else:
                    final = total
                paths = []
                images = []
                classes = []
                j = 1
        except BaseException as ex:
            print(ex)
            paths = []
            images = []
            classes = []
            j = 1
            
    # generate the logging csv
    final = pandas.DataFrame(final)
    final.columns = ['文件名', '类别id', '得分', '类别']
    final = final.set_index('文件名')
    final.to_csv(csv_path, encoding='utf_8_sig')
    print("predict finish==================================================================")


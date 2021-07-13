'''
import
'''
import pandas as pd
from model.EFF import *
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
)

'''
basic parameter config
'''
# define the img shape that we want to feed to the model
img_width = 480
img_height = 480

"""
the following train function is for training the CLS model
Parameters:
1. save_dir: the checkpoint you want to save, please use the format like '/home/zhang.xinxi/CV/checkpoint/man_top_color/cp-{epoch:04d}.ckpt' 
   so that the function can save different epochs.
2. csv_dir: the dir for logging the training history(e.g. acc/loss), end with '.csv'.
3. data_dir: the dir for the data, we use the tf api to load the data information(including the class information),so the data dir should be 
   structured(data structure: root path/class_name/images), and we use the root as the data_dir here.
4. best_path: since the we use the early stop mechanism here, we store the best epoch of the model to the best_path
5. epochs: the max epochs to train the model
6. portion: the portion for the training set -> (1-portion) is the the portion for the val set 
7. log_dir: the class information of the data, (e.g. '方领','v领','圆领'), end with '.csv'
8. begin_epoch: if you want to continue training using a certain checkpoint, you can enter the checkpoint here for the model to load
   the checkpoint must be ended with '.ckpt'
9. distribute_training: whether to use multiple GPU to train 
"""

def train(save_dir, csv_dir, data_dir, best_path, epochs, batch_size, portion,log_dir,begin_epoch='no',distribute_training=False):
    """
    call backs
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    logger = keras.callbacks.CSVLogger(csv_dir, separator=',', append=False)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_dir,
        verbose=1,
        save_weights_only=True,
        period=1)

    '''
    load the data
    '''
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1 - portion,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=1 - portion,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    class_names = train_ds.class_names
    class_nums = len(train_ds.class_names)
    print('the data has been loaded', train_ds.class_names)

    # transfer the label to one_hot form
    def map_func(images, label):
        one_hot_label = tf.one_hot(label, depth=class_nums)
        return images, one_hot_label

    train_ds = train_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)

    '''
    load the EFF model
    '''
    if distribute_training:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = get_EFFmodel(img_height, img_width, class_nums)
            if 'ckpt' in begin_epoch:
                model.load_weights(begin_epoch)
            model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    else:
        model = get_EFFmodel(img_height, img_width, class_nums)
        if 'ckpt' in begin_epoch:
            model.load_weights(begin_epoch)
        model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                      loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    print(model.summary())

    '''
    model training
    '''
    # prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # train
    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[cp_callback,
                        early_stop,
                        logger,
                        ReduceLROnPlateau(verbose=1)]
        )

    # save the best weights of the model
    model.save_weights(best_path)
    print("training finish==================================================================================")

    # save the class information to the log_dir
    names_df = pd.DataFrame(class_names).set_index(0)
    names_df.to_csv(log_dir)



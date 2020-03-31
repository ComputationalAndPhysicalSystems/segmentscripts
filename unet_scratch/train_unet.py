from unet_model import *
from gen_patches import *

import glob
import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 3
N_CLASSES = 5  # BG PR PP PY O
CLASS_WEIGHTS = [0.1,0.1, 0.3, 0.3, 0.25]
N_EPOCHS = 20
UPCONV = False
PATCH_SZ = 64    # should divide by 16
BATCH_SIZE = 10
TRAIN_SZ = 5000  # train size
VAL_SZ = 1000    # validation size
TRAIN_DIR = "/home/lpe/Desktop/junk/dish_ilastik_34/200_files/u_net_training_bicc_files/"
TEST_DIR = "/home/lpe/Desktop/junk/dish_ilastik_34/200_files/upsize/"


sorted_dir_listttt = glob.glob(TEST_DIR + "*.npy*")
sorted_dir_list0 = sorted(sorted_dir_listttt, key=lambda item: (int(item.partition('-')[1]) if item[1].isdigit() else float('inf'), item))#

sorted_dir_listttt = glob.glob(TRAIN_DIR + "*.tiff*")
sorted_dir_list1 = sorted(sorted_dir_listttt, key=lambda item: (int(item.partition('-')[1]) if item[1].isdigit() else float('inf'), item))#


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


#weights_path = 'weights'
#if not os.path.exists(weights_path):
#    os.makedirs(weights_path)
weights_path = '/mnt/lfs2/epst0545/deep-unet-for-satellite-image-segmentation/weights/funky_faf_1.hdf5'

trainIds = [str(i).zfill(2) for i in range(1, 200)]  # all availiable ids: from "01" to "24"

numpy_listinds = 0

if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:
	
        img_m = normalize(tiff.imread(sorted_dir_list1[numpy_listinds]))
	#mask = tiff.imread('/mnt/lfs2/epst0545/u_net_training/train_2/{}.tif'.format(img_id)).transpose([1, 2, 0]) 
        mask = np.load(sorted_dir_list0[numpy_listinds])#.transpose([1, 2, 0])
        numpy_listinds += 1

        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
       	print(x_train.shape,x_val.shape)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()

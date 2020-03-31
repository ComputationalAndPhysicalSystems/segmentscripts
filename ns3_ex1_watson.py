import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import tensorflow as tf
import glob
import cv2
from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES
from keras.models import load_model
from keras import backend as K
class_weights=[0.2, 0.2, 0.2, 0.2, 0.2]
print(tf.keras.backend.image_data_format())

np.random.seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

def weighted_binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses * K.constant(class_weights))

def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def predict(x, model, patch_sz=64, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    print("vert", npatches_vertical, "horizontal",npatches_horizontal) 
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    k = 0 
    for i in range(0, npatches_vertical):
        
        #i = k // npatches_horizontal
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        
        for j in range(0, npatches_horizontal):
        
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            #print(x0,x1,y0,y1)
            prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
            k += 1
            #print(k)
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0.4):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
	#print(i)
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict
import os
#currentDirectory = os.getcwd()
raw_dir = ("/mnt/lfs2/epst0545/" + '/segment_me/')
seg_dir = ("/mnt/lfs2/epst0545/" + '/segmented/')


#raw_dir = raw_dir[7:]
#seg_dir = seg_dir[7:]


new_dirs = os.listdir(raw_dir)

chunked = np.array_split(new_dirs, 4)
new_dirs2 = chunked[3]
for the_dirs in new_dirs2:
    TEST_DIR = (raw_dir + the_dirs + "/")

    trash_man = (seg_dir + the_dirs + "/")

    sorted_dir_listttt = glob.glob(TEST_DIR + "*.png*")
    sorted_dir_list0 = sorted(sorted_dir_listttt, key=lambda item: (int(item.partition('-')[1]) if item[1].isdigit() else float('inf'), item))#
    #sorted_dir_list0 = sorted_dir_list0[918:]
    #for z in sorted_dir_list0:

    if __name__ == '__main__':
         model = get_model()
         model.load_weights(weights_path)
         #model = load_model('/home/lpe/Desktop/junk/dish_ilastik_34/output/tha_gewd_u_net_2.h5',custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
         #test_id = "01"
         #img = normalize(tiff.imread(z)).transpose([1,0,2])  # make channels last

    for z in sorted_dir_list0:
           # img = normalize(tiff.imread(z))#.transpose([1,0,2])
            #img = image_resize(img,height=1173)#.transpose([1,0,2])
            print(z) 
            img = normalize(image_resize(cv2.imread(z), height = 1173))
            for i in range(7):
                if i == 0:  # reverse first dimension
                    mymat = predict(img[::-1,:,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([0,2,1])
                    #print(mymat[0][0][0], mymat[3][12][13])
                    print("Case 1",img.shape, mymat.shape)
                elif i == 1:    # reverse second dimension
                    temp = predict(img[:,::-1,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([1,0,2])
                    #print(temp[0][0][0], temp[3][12][13])
                    print("Case 2", temp.shape, mymat.shape)
                    mymat = np.median( np.array([ temp[:,::-1,:], mymat.transpose([2,0,1])]), axis=0 )
                elif i == 2:    # transpose(interchange) first and second dimensions
                    temp = predict(img.transpose([0,1,2]), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([0,2,1])
                    #print(temp[0][0][0], temp[3][12][13])
                    print("Case 3", temp.shape, mymat.shape)
                    mymat = np.median( np.array([ temp.transpose(2,0,1), mymat ]), axis=0 )
                elif i == 3:
                    #print(np.rot90(img, 1).shape)
                    temp = predict(np.rot90(img.transpose([1,0,2]), 1), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
                    #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
                    print("Case 4", temp.shape, mymat.shape)
                    mymat = np.median( np.array([ np.rot90(temp, -1).transpose([0,1,2]), mymat ]), axis=0 )
                elif i == 4:
                    temp = predict(np.rot90(img.transpose([0,1,2]), 2), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
                    #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
                    print("Case 5", temp.shape, mymat.shape)
                    mymat = np.median( np.array([ np.rot90(temp,-2).transpose([1,0,2]), mymat ]), axis=0 )
                elif i == 5:
                    temp = predict(np.rot90(img.transpose([1,0,2]), 3), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
                    #print(temp.transpose([2,0,1])[0][0][0], temp.transpose([2,0,1])[3][12][13])
                    print("Case 6", temp.shape, mymat.shape)
                    mymat = np.median( np.array([ np.rot90(temp, -3).transpose(0,1,2), mymat ]), axis=0 )
                else:
                    temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
                    #pnspose([1,0,2])rint(temp[0][0][0], temp[3][12][13])
                    print("Case 7", temp.shape, mymat.shape,(mymat.transpose([2,0,1])).shape)
                    mymat = np.median( np.array([ temp, mymat.transpose([2,1,0]) ]), axis=0 )

            #print(mymat[0][0][0], mymat[3][12][13])
            map = picture_from_mask(mymat, 0.5)
            #mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])  # make channels first
            #map = picture_from_mask(mask, 0.5)

            #tiff.imsave('result.tif'r (255*mask).astype('uint8'))

            #tiff.imsave('result.tif', (255*mymat).astype('uint8'))
            ayo = z.split("/")[-1]
            #print(ayo[:-5])
            thaboy = ayo[:-5]
            thaboy = trash_man + thaboy
            segname = (thaboy + "_unet_segmented" +".tiff")
            tiff.imsave(segname, map)

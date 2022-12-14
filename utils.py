from PIL import Image
import PIL.ImageFont, PIL.ImageDraw
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

def give_color_to_annotation(annotation):
    '''
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
        annotation (numpy array) - label map array
    
    Returns:
        the annotation array with an additional color channel/axis
    '''

    class_names = ['background','crack']
    # generate a list that contains one color for each class
    colors = sns.color_palette(None, len(class_names))

    # print class name - normalized RGB tuple pairs
    # the tuple values will be multiplied by 255 in the helper functions later
    # to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
    for class_name, color in zip(class_names, colors):
        print(f'{class_name} -- {color}')

    seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
    #print(annotation)

    for c in range(2):
        segc = (annotation == c)
        seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
        seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
        seg_img[:,:,2] += segc*( colors[c][2] * 255.0)

    return seg_img

def show_single_predictions(model, filename):
    pred_img = predict_image(model, filename)
    plt.imshow(pred_img)

def predict_image(model, filename):
    '''
    Displays the images with the ground truth and predicted label maps

    Args:
        image (numpy array) -- the input image
        labelmaps (list of arrays) -- contains the predicted and ground truth label maps
        titles (list of strings) -- display headings for the images to be displayed
        iou_list (list of floats) -- the IOU values for each class
        dice_score_list (list of floats) -- the Dice Score for each vlass
    '''
    height = 224
    width = 224

    # img_raw = tf.io.read_file(filename)
    # image = tf.image.decode_jpeg(img_raw)

    # # Resize image and segmentation mask
    # image = tf.image.resize(image, (height, width,))
    # image = tf.reshape(image, (height, width, 3,))

    # image = image/255

    # img_array = tf.expand_dims(image, 0) # Create a batch

    image = Image.open(filename).resize((width, height))

    image = np.array(image)/255

    img_array = np.expand_dims(image, axis=0)

    pred_imgs = model.predict(img_array)
    pred_imgs = np.argmax(pred_imgs, axis=3)
    
    pred_img = give_color_to_annotation(pred_imgs[0])
    pred_img = np.uint8(pred_img)
    return pred_img
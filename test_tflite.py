#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import seaborn as sns
import time
from PIL import Image
from matplotlib import pyplot as plt


class_names = ['background','crack']

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))

# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
for class_name, color in zip(class_names, colors):
  print(f'{class_name} -- {color}')

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
  seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
  #print(annotation)
  
  for c in range(2):
    segc = (annotation == c)
    seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
    seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
    seg_img[:,:,2] += segc*( colors[c][2] * 255.0)
  
  return seg_img


TFLITE_FILE_PATH = "models/model.tflite"
# Load the TFLite model in TFLite Interpreter
#interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter = tflite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
#my_signature = interpreter.get_signature_runner()
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# my_signature is callable with input as arguments.
#output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
IMG_PATH = "test_imgs/CFD_003.jpg"
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# img_raw = tf.io.read_file(IMG_PATH)
# image = tf.image.decode_jpeg(img_raw)

# # Resize image and segmentation mask
# image = tf.image.resize(image, (height, width,))
# image = tf.reshape(image, (height, width, 3,))

image = Image.open(IMG_PATH).resize((width, height))

# Normalize pixels in the input image
image = np.array(image, dtype='float32')/255
#image -= 1

input_data = np.expand_dims(image, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
stop_time = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])
pred_imgs = np.argmax(output_data, axis=3)
#result = np.squeeze(output_data)
pred_img = give_color_to_annotation(pred_imgs[0])
pred_img = np.uint8(pred_img)
Image.fromarray(pred_img, 'RGB').show()
#results = np.squeeze(output_data)

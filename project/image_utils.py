import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#apply resnet50 preprocessing and then apply resnet to the image
#TODO: maybe cache these features to disk using a preprocessing script?
def extract_features(images):
    resnet = ResNet50(include_top=False)
    images = images.map(preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
    images = images.map(resnet, num_parallel_calls=tf.data.AUTOTUNE)
    return images

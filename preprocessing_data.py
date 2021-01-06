
# Import Libraries
import dlib
import glob
import cv2
import os
import sys
import  time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil


directory = 'train_images_h(change programe)'
box_file = 'boxes_h.txt(change programe)'
# In this dictionary our images and annotations will be stored.
data = {}

# Get the indexes of all images.
image_indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(directory)]

# Shuffle the indexes to have random train/test split later on.
np.random.shuffle(image_indexes)

# Open and read the content of the boxes.txt file
f = open(box_file, "r")
box_content = f.read()

# Convert the bounding boxes to dictionary in the format `index: (x1,y1,x2,y2)` ...
box_dict = eval('{' + box_content + '}')

# Close the file
f.close()

# Loop over all indexes
for index in image_indexes:
    # Read the image in memmory and append it to the list
    img = cv2.imread(os.path.join(directory, str(index) + '.png'))

    # Read the associated bounding_box
    bounding_box = box_dict[index]

    # Convert the bounding box to dlib format
    x1, y1, x2, y2 = bounding_box
    dlib_box = [dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)]

    # Store the image and the box together
    data[index] = (img, dlib_box)

print('Number of Images and Boxes Present: {}'.format(len(data)))
no_of_samples = 10

image_names = os.listdir(directory)

np.random.shuffle(data)

# Extract the subset of boxes
# subset = data[][:no_of_samples ]

cols = 5

# Given the number of samples to display, what's the number of rows required.
rows = int(np.ceil(no_of_samples / cols))

# Set the figure size
plt.figure(figsize=(cols * cols, rows * cols))

# Loop for each class
for i in range(no_of_samples):
    # Extract the bonding box coordinates
    d_box = data[i][1][0]
    left, top, right, bottom = d_box.left(), d_box.top(), d_box.right(), d_box.bottom()

    # Get the image
    image = data[i][0]

    # Draw reectangle on the detected hand
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    # Display the image
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image[:, :, ::-1])
    plt.axis('off')
# This is the percentage of data we will use to train
# The rest will be used for testing
percent = 0.8

# How many examples make 80%.
split = int(len(data) * percent)

# Seperate the images and bounding boxes in different lists.
images = [tuple_value[0] for tuple_value in data.values()]
bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

# Initialize object detector Options
options = dlib.simple_object_detector_training_options()

# I'm disabling the horizontal flipping, becauase it confuses the detector if you're training on few examples
# By doing this the detector will only detect left or right hand (whichever you trained on).
options.add_left_right_image_flips = False
# Set the c pa rameter of SVM equal to 5
# A bigger C encourages the model to better fit the training data, it can lead to overfitting.
# So set an optimal C value via trail and error.
options.C = 8

# Note the start time before training.
st = time.time()

# You can start the training now
detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

# Print the Total time taken to train the detector
print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

file_name = 'change_programe.svm'
detector.save(file_name)



print("Training Metrics: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))
detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
detector.save(file_name)

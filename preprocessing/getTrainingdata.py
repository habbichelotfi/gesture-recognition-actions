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

# If cleanup is True then the new images and annotations will be appended to previous ones
# If False then all previous images and annotations will be deleted.
cleanup = True

# Set the window to a normal one so we can adjust it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Resize the window and adjust it to the center
# This is done so we're ready for capturing the images.
cv2.resizeWindow('frame', 1920, 1080)
cv2.moveWindow("frame", 0, 0)

# Initialize webcam
cap = cv2.VideoCapture(1)

# Initalize sliding window's x1,y1
x1, y1 = 0, 0

# These will be the width and height of the sliding window.
window_width = 190  # 140
window_height = 190

# We will save images after every 4 frames
# This is done so we don't have lot's of duplicate images
skip_frames = 3
frame_gap = 0

# This is the directory where our images will be stored
# Make sure to change both names if you're saving a different Detector
directory = 'train_images_h(change programe)'
box_file = 'boxes_h.txt(change programe)'

# If cleanup is True then delete all imaages and bounding_box annotations.
if cleanup:

    # Delete the images directory if it exists
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # Clear up all previous bounding boxes
    open(box_file, 'w').close()

    # Initialize the counter to 0
    counter = 0

elif os.path.exists(box_file):

    # If cleanup is false then we must append the new boxes with the old
    with open(box_file, 'r') as text_file:
        box_content = text_file.read()

    # Set the counter to the previous highest checkpoint
    counter = int(box_content.split(':')[-2].split(',')[-1])

# Open up this text file or create it if it does not exists
fr = open(box_file, 'a')

# Create our image directory if it does not exists.
if not os.path.exists(directory):
    os.mkdir(directory)

# Initial wait before you start recording each row
initial_wait = 0

# Start the loop for the sliding window
while (True):

    # Start reading from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Invert the image laterally to get the mirror reflection.
    frame = cv2.flip(frame, 1)

    # Make a copy of the original frame
    orig = frame.copy()

    # Wait the first 50 frames so that you can place your hand correctly
    if initial_wait > 60:

        # Increment frame_gap by 1.
        frame_gap += 1

        # Move the window to the right by some amount in each iteration.
        if x1 + window_width < frame.shape[1]:
            x1 += 4
            time.sleep(0.1)

        elif y1 + window_height + 270 < frame.shape[1]:

            # If the sliding_window has reached the end of the row then move down by some amount.
            # Also start the window from start of the row
            y1 += 80
            x1 = 0

            # Setting frame_gap and init_wait to 0.
            # This is done so that the user has the time to place the hand correctly
            # in the next row before image is saved.
            frame_gap = 0
            initial_wait = 0

        # Break the loop if we have gone over the whole screen.
        else:
            break

    else:
        initial_wait += 1

    # Save the image every nth frame.
    if frame_gap == skip_frames:
        # Set the image name equal to the counter value
        img_name = str(counter) + '.png'

        # Save the Image in the defined directory
        img_full_name = directory + '/' + str(counter) + '.png'
        cv2.imwrite(img_full_name, orig)

        # Save the bounding box coordinates in the text file.
        fr.write('{}:({},{},{},{}),'.format(counter, x1, y1, x1 + window_width, y1 + window_height))

        # Increment the counter
        counter += 1

        # Set the frame_gap back to 0.
        frame_gap = 0

    # Draw the sliding window
    cv2.rectangle(frame, (x1, y1), (x1 + window_width, y1 + window_height), (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and close the file and window
cap.release()
cv2.destroyAllWindows()
fr.close()
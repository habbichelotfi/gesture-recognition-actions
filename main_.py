
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

file_name="models/Pause_detector.svm"
pause_detector = dlib.fhog_object_detector(file_name)
scrolling_detector=dlib.fhog_object_detector("models/Scrolling_up.svm")
scrolling_tabs_detector=dlib.fhog_object_detector("models/Scrolling_tabs.svm")
change_program=dlib.fhog_object_detector("models/change_programe.svm")
detectors=[pause_detector,scrolling_detector,scrolling_tabs_detector,change_program]
names = ['Pause Detected', 'Scrolling Detected','Scrolling TABS','Change programme']

cap=cv2.VideoCapture(1)

scale_factor = 4
size, center_x = 0,0

fps = 0
frame_counter = 0
start_time = time.time()

while True:

    _,frame=cap.read()
    if not _:
        break
    frame=cv2.flip(frame,1)
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    copy=frame.copy()
    new_width = int(frame.shape[1] / scale_factor)
    new_height=int(frame.shape[0]/scale_factor)
    resized_frame=cv2.resize(frame,(new_width,new_height))

    [detections, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, resized_frame,
                                                                                      upsample_num_times=1,adjust_threshold=0.0)

    for i in range(len(detections)):
        x1=int(detections[i].left()*scale_factor)
        y1=int(detections[i].top()*scale_factor)
        x2=int(detections[i].right()*scale_factor)
        y2=int(detections[i].bottom()*scale_factor)


        if confidences[i]*100>90.0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,'{}: {:.2f}%'.format(names[detector_idxs[i]],confidences[i]*100),(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)
        size = int((x2 - x1) * (y2 - y1))
        #print(type(confidences[i]))
        # Extraqct the center of the hand on x-axis.
        center_x = x2 - x1 // 2
        if names[detector_idxs[i]]=='Pause Detected' and confidences[i]*100>90.0:
            pyg.press('space')
            cv2.waitKey(500)
        if names[detector_idxs[i]]=='Scrolling Detected' and (confidences[i]*100)>90.0:
            pyg.scroll(-7)
        if names[detector_idxs[i]]=='Scrolling TABS' and (confidences[i]*100)>90.0:
            pyg.hotkey('ctrl', 'pgup')
            cv2.waitKey(500)
        if names[detector_idxs[i]]=='Change programme' and (confidences[i]*100)>90.0:
            pyg.hotkey('alt', 'tab')

        # Display FPS and size of hand
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    # This information is useful for when you'll be building hand gesture applications
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))

    cv2.imshow("a",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
cap.release()
cv2.destroyAllWindows()
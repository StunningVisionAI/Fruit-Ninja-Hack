'''
Fruit Ninja Hack 1.0

Created By : Priyanto Hidayatullah and Stunning Vision AI Team
(C) 2022

'''
import time
import cv2
import mss
import numpy as np
import os
import win32api, win32con
import sys
import math
import mouse
import keyboard

# YOLOv4
CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.5
COLORS = [(0,255,255), (255,255,0), (255,0,0)]

class_names = []

with open("data/fruit.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("weights/yolov4-fruit-v2_best.weights", "cfg/yolov4-fruit-v2.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

screenHeight = win32api.GetSystemMetrics(1)
screenWidth = win32api.GetSystemMetrics(0)

# Resolution Game
width = 960
height = 540

# Top Corner
gameScreen = {'top': 0, 'left': screenWidth - width, 'width': width, 'height': height}


def checkBomb(bombs_path, object_path, object_width, object_height):

    for bomb_path in bombs_path:
        width_range = abs(bomb_path[0] - object_path[0])
        height_range = abs(bomb_path[1] - object_path[1])

        if(height_range <= (object_height * 2.25)):
            if((width_range == object_width) or (width_range <= (object_width/2))):            
                return True

    return False

def slice(x, y, height):
    mouse.move(int(x), int(y), absolute=True)
    mouse.drag(0, 0, 0, -(int(height * 2.25)), absolute=False, duration=0.025)

def getRealCoord(center, y, heightObject):
    return (center + (1920 - width), y + heightObject)


with mss.mss() as sct:
    while True:
        
        # Screenshot
        screen = np.array(sct.grab(gameScreen))
        screen = np.flip(screen[:, :, :3], 2) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        img = screen.copy()        

        # Object Detection
        start = time.time()
        clasess, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()
        
        bombs_path = []
        fruits_path = []

        for (classid, score, box)  in zip(clasess, scores, boxes):
            # Object Color
            color = COLORS[int(classid) % len(COLORS)]
            # Object Label
            label = "%s : %f" % (class_names[classid[0]], score)

            # Center Object
            center = box[0] + int(box[2]/2)            
            start_point = getRealCoord(center, box[1], box[3])                           

            # Draw Box Object Detection
            cv2.rectangle(img, box, color, 2)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Bomb
            if((classid[0] == 1)):
                bombs_path.append(start_point);

            # Fruit           
            if((classid[0] == 0)):
                fruits_path.append(box);

        # Slice Detected Object
        for fruit_path in fruits_path:
            center = fruit_path[0] + int(fruit_path[2]/2)
            start_point = getRealCoord(center, fruit_path[1], fruit_path[3])
            nearBomb = checkBomb(bombs_path, start_point, fruit_path[2], fruit_path[3])            

            if(not nearBomb):
                slice(start_point[0], start_point[1], fruit_path[3])
        
        # Draw Result
        fps_label = "FPS: %.2f" % (1 / (end - start))
        cv2.putText(img, fps_label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)   
        cv2.imshow('img', img)

        cv2.waitKey(1)
        # Press 'q' to quit
        if keyboard.is_pressed('q'):
            cv2.destroyAllWindows()
            quit()
quit()
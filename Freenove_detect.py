# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from Vignesh_FreeNova_Test import StartCar,KeepRunningTillObstacle,TurnRight,TurnLeft
from servo import *
pwm=Servo()
from Buzzer import *
buzzer=Buzzer()
from Motor import *            
PWM=Motor()   
from Ultrasonic import *
ultrasonic=Ultrasonic() 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  
  pipeline = "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    #result_text = ""
    # Draw keypoints and edges on input image
    image, binary_array, result_text  = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    
    data=ultrasonic.get_distance()   #Get the value
    printText = " Obstacle distance is "+str(data)+"CM"
    printText = ""
    fps_text = 'FPS = {:.1f}'.format(fps) + printText
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    
    
    
    #Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      PWM.setMotorModel(0,0,0,0)
      break
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # Move the window to the left side of the screen (x=0, y=0)    
    cv2.resizeWindow('image', 640, 480)
    
    cv2.moveWindow('image', 1000, 1000)
    # Display the image
    cv2.imshow('image', image)    
    
    cv2.waitKey(2000)
    
    	
    
    cmap = ListedColormap(['green', 'red'])
    plt.figure(figsize=(6, 6))  # Set the figure size for better visibility
    plt.imshow(binary_array, cmap)  # Display in grayscale
    plt.title(fps_text, fontsize=16)  # Add a title
    plt.axis('off')  # Turn off the axis for a cleaner display        
    plt.show(block=False)
    plt.pause(2)    
    plt.close()
    cap.release()
    cv2.destroyAllWindows()
    
    return binary_array , data, result_text


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  return run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


def OpenCVScanner():
  i=30
  
  Map = {}
  objects ={}
  uSense = {}
  result_text = ""
  while(i<=150):
    PWM.setMotorModel(0,0,0,0)
    pwm.setServoPwm('1',90)     
    pwm.setServoPwm('0',i)     
    time.sleep(1)
    try:
      score, data, result_text = main() 
    except Exception as e:
      print(e)
    Map[i] = score
    objects[i] = result_text
    uSense[i] = data
    i = i +60
  
  all_arrays = list(Map.values())
  result = np.concatenate(all_arrays, axis=1) 
  cmap = ListedColormap(['green', 'red','blue'])
  plt.figure(figsize=(12, 6))  # Set the figure size for better visibility
  plt.imshow(result, cmap)  # Display in grayscale
  plt.title("Combined MAP", fontsize=16)  # Add a title
  plt.axis('off')  # Turn off the axis for a cleaner display        
  plt.show(block=False)
  plt.pause(2)      
  plt.close()
  print("Map Closed")

  
  motorSpeed = 800
  sleepTime = motorSpeed/15
  for key, value in objects.items():
    print(value)
    if 'stop' in value:
      print("Stop found")        
      PWM.setMotorModel(motorSpeed,motorSpeed,motorSpeed,motorSpeed)  
      print(uSense[key])  
      time.sleep(uSense[key]/sleepTime)
      PWM.setMotorModel(0,0,0,0)
      time.sleep(3)
  
  CountSteps(result)
  pwm.setServoPwm('0',90) 
  GreenMapDict = {key: np.count_nonzero(value == 0) for key, value in Map.items()}
  max_key = max(GreenMapDict, key=GreenMapDict.get)
  if(max_key<90):
    TurnLeft()
    PWM.setMotorModel(motorSpeed,motorSpeed,motorSpeed,motorSpeed)        
    time.sleep(1)
    PWM.setMotorModel(0,0,0,0)
    TurnRight()
    #KeepRunningTillObstacle()  
  if(max_key>90):
    TurnRight()
    PWM.setMotorModel(motorSpeed,motorSpeed,motorSpeed,motorSpeed)        
    time.sleep(1)
    PWM.setMotorModel(0,0,0,0)
    TurnLeft()
    #KeepRunningTillObstacle()
  pwm.setServoPwm('0',90) 
  KeepRunningTillObstacle()    
  
  

def CountSteps(binary_map):
  # Example 1920x1140 binary map (replace with your actual map)
  height,width = binary_map.shape    

  # Start at the bottom middle of the array
  start_row = height - 1
  start_col = width // 2

  # Initialize variables
  current_row = start_row
  current_col = start_col
  up_steps = 0
  left_turns = 0
  right_turns = 0

  # Move up until you hit an obstacle or reach the top of the map
  while current_row >= 0:
      if binary_map[current_row, current_col] == 1:  # Obstacle encountered
          # Check left and right for free space (0)
          if current_col > 0 and binary_map[current_row, current_col - 1] == 1:  # Turn left
              current_col -= 1
              left_turns += 1          
          elif current_col < width - 1 and binary_map[current_row, current_col + 1] == 0:  # Turn right
              current_col += 1
              right_turns += 1
          else:
              # No free space to turn; stop moving
              break
      else:
          # Move up if no obstacle is encountered
          current_row -= 1
          up_steps += 1

  # Output the results
  print(f"Up steps: {up_steps}")
  print(f"Left turns: {left_turns}")
  print(f"Right turns: {right_turns}")
  return up_steps,left_turns,right_turns



if __name__ == '__main__':   
  try: 
    OpenCVScanner()
  except:
    PWM.setMotorModel(0,0,0,0)
  
  
         

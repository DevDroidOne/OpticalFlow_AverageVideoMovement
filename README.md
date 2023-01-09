# OpticalFlow_AverageVideoMovement
This was part of my Honours project for computer science.
## Components
The components for the project consisted of a raspberry pi a motor driver a small cheap web camera and a motor itself the part bringing it together.

## Library's

The library's used make it possible to find and track flow accords a image with selected points.
As seen everything is written in python and not in the most efficient manner but the project was used a proof of concept.
 - Opencv
 - Scipy
 - Math
 
# Behavior
 
 When starting the program will search or a camera connected to the system make sure U have the necessary libraries installed with PIP and the newest version of python
 When the video is running certain points will be chosen to be tracked marked by the red dots and the blue boxes around it is the size of kernel used.
 
The point right in the middle of the video is the average estimated movement of the video. The optical flow method used is the luckas kanade method. 




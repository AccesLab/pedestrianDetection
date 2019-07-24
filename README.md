"@author 1: mm_islam

@author 2: Tgtadewos

Supervisor: Dr. Ali Karimoddini

# This modified ROS node provides the specific coordinates, approximate distance, and specific size of the detected pedestrians. By partitioning, the image into three parts the region of interest (left, right, or middle) for each pedestrian is reported. Also, based on the size of the detection window, the approximate distance from a pedestrian in the field of view of the camera is estimated.

# This code can be used with other trained models as well. 

# To validate the developed model, the code has been tested via on-campus and off-campus autonomous driving facilities.

#This effort is supervised by Dr. Ali Karimoddni and supported by STATE OF NORTH CAROLINA, DEPARTMENT OF TRANSPORTATION under the project number 2019-28, led by the Institute for Transportation Research and Education (ITRE) at NC State University and Co-led by Autonomous Cooperative Control of Emergent Systems of Systems (ACCESS) Laboratory at NC A&T State University.

#This code is a modification of multiple object detection with SSD_mobilenet V1.

# The original code is from https://github.com/osrf/tensorflow_object_detector/blob/master/README.md.

# We customized the code to enforce and limit object detection to identify only pedestrians.
# steps to run SSD (Single Shot Detection):


Step 1: Install ROS: http://wiki.ros.org/kinetic/Installation/Ubuntu

Step 2: Install camera dependencies based on the camera that you are using

Step 3: Install tensorflow into python virtualenv: https://www.tensorflow.org/install/install_linux

    sudo apt-get install python-pip python-dev python-virtualenv

    virtualenv --system-site-packages ~/tensorflow

    source ~/tensorflow/bin/activate

    easy_install -U pip

    pip install --upgrade tensorflow
You can skip that step, if your machine already have the setup.
Step 4: mkdir ~/catkin_ws/ && mkdir ~/catkin_ws/src/    
Step 5: Clone standard Vision messages repository and this repository into catkin_ws/src:

    cd ~/catkin_ws/src

    git clone https://github.com/Kukanani/vision_msgs.git

    git clone https://github.com/osrf/tensorflow_object_detector.git
    
Step 6:   Build tensorflow_object_detector and Vision message

    cd ~/catkin_ws && catkin_make
Step 7: 
    Source catkin workspace's setup.bash:

    source ~/catkin_ws/devel/setup.bash  
Step 8: Plug in camera and launch Single Shot Detector (varies per camera, NOTE: object_detect.launch also launches the openni2.launch file for the camera. If you are using any other camera, please change the camera topic in the launch file before launching the file)

    roslaunch tensorflow_object_detector object_detect.launch

    OR

    roslaunch tensorflow_object_detector usb_cam_detector.launch

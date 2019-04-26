# Smart Aquarium
Fish 3D tracking and abnormal behavior detection.


## Requirements
- Ubuntu 16.04
- Python 3.5
- TensorFlow 1.9
- OpenCV 3.4


## Getting Started
Creating virtualenv
```bash
$ cd Smart-Aquarium
$ virtualenv env --python=python3.5
$ source env/bin/activate
```

Install Dependencies
```bash
$ sudo apt-get install mysql-server
$ sudo apt-get install python3-tk
$ pip install -r requirements.txt
```

Run
```bash
$ python detector.py [input device]

  1. Using USB CAMERA
    python detector.py camera
    
  2. Using VIDEO
    python detector.py video
    
  3. Using RealSense
    python detector.py realsense
````

Monitoring System
```bash
$ python monitoring.py
````


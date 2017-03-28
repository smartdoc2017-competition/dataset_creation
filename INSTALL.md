ICDAR 2017 Competition SmartDoc-reconstruction
==============================================
==============================================

Installation procedure for dataset creation tools
=================================================

Dependencies
------------

OpenCV
First make sure you have OpenCV 2.8+ installed with Python 2.7+ support.
OpenCV 3.x is not supported.
If you use a virtual environment (we recommend doing so, see 
http://virtualenvwrapper.readthedocs.io to get started) then you should use the
--system-site-packages flag to make sure your environment can use OpenCV.

Numpy
The second and last requirement is Numpy.
You can call any of the following commands:
  $ pip install -U numpy
or
  $ pip install -r requirements.txt

Installation
------------
Just copy the files of this archive, no dedicated installation required.



Usage
=====

Here is a sample usage:

  $ python create_reference.py -d \
      /path/to/dataset/screen01/ground-truth.png \
      /path/to/dataset/screen01/input.mp4 \
      /path/to/dataset/screen01

You can call
  $ python create_reference.py -d
to read the online help.


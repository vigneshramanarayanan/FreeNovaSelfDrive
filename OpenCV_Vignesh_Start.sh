export PYTHONPATH=/usr/local/lib/python3.9/site-packages:$PYTHONPATH
source ~/.bashrc
cd /home/vignesh/examples/lite/examples/object_detection/raspberry_pi
python3 Freenove_detect.py \
  --model efficientdet_lite0.tflite

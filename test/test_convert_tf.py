import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf
import os
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel

# from IPython.display import Audio
# print(tf.__version__)
os.chdir("..")
import sys
sys.path.append("MyTFTTS/")
tacotron2 = TFAutoModel.from_pretrained("bert-base-chinese", enable_tflite_convertible=True)
tacotron2.setup_window(win_front=6, win_back=6)
tacotron2.setup_maximum_iterations(3000)

tacotron2.summary()
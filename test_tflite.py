import numpy as np
import yaml
import tensorflow as tf

import sys
sys.path.append('/home/TensorFlowTTS')

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel

from IPython.display import Audio
melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")
interpreter = tf.lite.Interpreter(model_path='fastspeech_quant.tflite')

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print(input_details)
#exit()
output_details = interpreter.get_output_details()

# Prepare input data.
def prepare_input(input_ids):
  input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
  return (input_ids,
          tf.convert_to_tensor([0], tf.int32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32))

# Test the model on random input data.
def infer(input_text):
  processor = AutoProcessor.from_pretrained(pretrained_path="/home/TensorFlowTTS/test/files/ljspeech_mapper.json")
  input_ids = processor.text_to_sequence(input_text.lower())
  interpreter.resize_tensor_input(input_details[0]['index'],
                                  [1, len(input_ids)])
  interpreter.resize_tensor_input(input_details[1]['index'],
                                  [1])
  interpreter.resize_tensor_input(input_details[2]['index'],
                                  [1])
  #interpreter.resize_tensor_input(input_details[3]['index'],[1])
  #interpreter.resize_tensor_input(input_details[4]['index'],[1])
  interpreter.allocate_tensors()
  input_data = prepare_input(input_ids)
  for i, detail in enumerate(input_details):
    input_shape = detail['shape_signature']
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))

input_text = "Recent research at Harvard has shown meditating\
for as little as 8 weeks, can actually increase the grey matter in the \
parts of the brain responsible for emotional regulation, and learning."

decoder_output_tflite, mel_output_tflite = infer(input_text)
audio_before_tflite = melgan(decoder_output_tflite)[0, :, 0]
audio_after_tflite = melgan(mel_output_tflite)[0, :, 0]

Audio(data=audio_before_tflite, rate=22050)
Audio(data=audio_after_tflite, rate=22050)

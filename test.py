import numpy as np
import yaml
import tensorflow as tf

import sys
sys.path.append('/home/TensorFlowTTS')

from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel

from IPython.display import Audio
print(tf.__version__) # check if >= 2.4.0

# initialize melgan model
melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")

# initialize FastSpeech model.
fastspeech = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech-ljspeech-en")

input_text = "Recent research at Harvard has shown meditating\
for as little as 8 weeks, can actually increase the grey matter in the \
parts of the brain responsible for emotional regulation, and learning."

processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech-ljspeech-en")
input_ids = processor.text_to_sequence(input_text.lower())

mel_before, mel_after, duration_outputs = fastspeech.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    # f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    # energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
)

audio_before = melgan(mel_before)[0, :, 0]
audio_after = melgan(mel_after)[0, :, 0]

Audio(data=audio_before, rate=22050)
Audio(data=audio_after, rate=22050)


# Concrete Function
fastspeech_concrete_function = fastspeech.inference_tflite.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [fastspeech_concrete_function]
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('fastspeech_quant.tflite', 'wb') as f:
  f.write(tflite_model)

print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0) )


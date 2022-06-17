import logging, os
logging.disable(logging.WARNING)
import os

import tensorflow_io as tfio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')



@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def main():

	reloaded_model = tf.saved_model.load('./cnn_model')

	files = [
	'./data/dog_1.wav',
	'./data/dog_2.wav',
	'./data/human_english.wav',
	'./data/human_english2.wav',
	'./data/engine_1.wav',
	'./data/engine_2.wav'
	]

	my_classes = ['insan', 'hayvan', 'arac', 'arkaplan']

	for file in files:
	    
	    testing_wav_data = load_wav_16k_mono(file)

	    reloaded_results = reloaded_model(testing_wav_data)
	    predicted = my_classes[tf.math.argmax(reloaded_results)]

	    print([file, predicted])

if __name__ == "__main__":
    main()


import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

# Load models
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
classifier = tf.keras.models.load_model('snoring_classifier.h5')

# Load long audio file
wav_file = 'm1.wav'
waveform, sr = librosa.load(wav_file, sr=16000)
duration_sec = librosa.get_duration(y=waveform, sr=sr)

# Parameters
chunk_size = 5  # seconds
chunk_samples = chunk_size * sr
snoring_chunks = 0

# Split and classify chunks
for i in range(0, len(waveform), chunk_samples):
    chunk = waveform[i:i+chunk_samples]
    if len(chunk) < chunk_samples:
        break  # skip short end

    # Convert to tensor
    chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)

    # Run through YAMNet
    scores, embeddings, spectrogram = yamnet_model(chunk_tensor)
    embedding = tf.reduce_mean(embeddings, axis=0)  # average across time

    # Predict snoring (0 or 1)
    pred = classifier.predict(np.expand_dims(embedding.numpy(), axis=0))
    if pred[0][0] > 0.5:
        snoring_chunks += 1

# Result
snoring_duration = snoring_chunks * chunk_size
print(f" Total snoring duration: {snoring_duration} seconds out of {int(duration_sec)} seconds.")

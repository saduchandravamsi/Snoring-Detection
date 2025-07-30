import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
import kagglehub
# Load models
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')



classifier = tf.keras.models.load_model('snoring_classifier.h5')

# Load and preprocess 1-second audio
audio_path = '1_489.wav'  # 1-sec audio file
waveform, sr = librosa.load(audio_path, sr=16000)

# Convert to tensor
waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

# Run through YAMNet to get embeddings (1024-dim)
_, embeddings, _ = yamnet_model(waveform_tensor)

# Average embeddings over time (mean across all time frames)
embedding_mean = tf.reduce_mean(embeddings, axis=0)

# Predict using custom classifier
prediction = classifier.predict(np.expand_dims(embedding_mean.numpy(), axis=0), verbose=0)

# Output
label = "Snoring" if prediction[0][0] > 0.5 else "Non-snoring"
print(f" Prediction: {label} (Confidence: {prediction[0][0]:.2f})")

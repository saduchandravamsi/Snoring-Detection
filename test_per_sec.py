import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

#  Load YAMNet from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print(" YAMNet model loaded.")

#  Load your snoring classifier (previously trained model)
classifier = tf.keras.models.load_model('snoring_classifier.h5')
print(" Snoring classifier model loaded.")

#  Load long audio file for testing
audio_path = 'm0.wav'  # Replace with your actual path
waveform, sr = librosa.load(audio_path, sr=16000)
total_seconds = int(librosa.get_duration(y=waveform, sr=sr))
print(f" Audio loaded: {total_seconds} seconds")

#  Chunk the audio per second and run prediction
chunk_duration = 1
chunk_samples = chunk_duration * sr
snoring_seconds = 0
timeline = []

print("🔍 Classifying audio second-by-second...")

for sec in range(total_seconds):
    start = sec * chunk_samples
    end = start + chunk_samples
    chunk = waveform[start:end]

    if len(chunk) < chunk_samples:
        break

    chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(chunk_tensor)
    embedding = tf.reduce_mean(embeddings, axis=0)

    prediction = classifier.predict(np.expand_dims(embedding.numpy(), axis=0), verbose=0)
    label = 1 if prediction[0][0] > 0.5 else 0
    timeline.append(label)

    if label == 1:
        snoring_seconds += 1

# Final snoring time
print(f"\n Total snoring time: {snoring_seconds} seconds out of {total_seconds}")

#  Visualization: Fuzzy-style plot
sns.set_style("whitegrid")
plt.figure(figsize=(15, 3))
plt.title('Snoring Timeline ')
plt.xlabel('Time (seconds)')
plt.ylabel('Snoring Prediction')

# 🎚️ Smooth the timeline using moving average
timeline_fuzzy = np.convolve(timeline, np.ones(3)/3, mode='same')

# 🟣 Plot smooth line
plt.plot(timeline_fuzzy, color='purple', linewidth=1.5, label='Smoothed Snoring Signal')

# 🔴 Fill snoring and 🔵 non-snoring
plt.fill_between(range(len(timeline_fuzzy)), 0, timeline_fuzzy, where=np.array(timeline_fuzzy) > 0.5, 
                 color='red', alpha=0.4, label='Snoring')
plt.fill_between(range(len(timeline_fuzzy)), 0, timeline_fuzzy, where=np.array(timeline_fuzzy) <= 0.5, 
                 color='blue', alpha=0.3, label='Non-snoring')

#  Visual tweaks
plt.ylim(0, 1.05)
plt.yticks([0, 0.5, 1], ['Non-snoring', 'Threshold', 'Snoring'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Load YAMNet from Kaggle Hub
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")
yamnet_model = tf.saved_model.load(path)
yamnet_infer = yamnet_model.signatures['serving_default']
print("✅ YAMNet model loaded from Kaggle Hub.")

# Load your snoring classifier
classifier = tf.keras.models.load_model('snoring_classifier.h5')
print("✅ Snoring classifier model loaded.")

# Load audio
audio_path = 'm0.wav'  # Replace with your actual path
waveform, sr = librosa.load(audio_path, sr=16000)
total_seconds = int(librosa.get_duration(y=waveform, sr=sr))
print(f"🎧 Audio loaded: {total_seconds} seconds")

# Analyze audio per second
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
    yam_output = yamnet_infer(waveform=chunk_tensor)
    embeddings = yam_output['output_1']  # 1024-dim embedding
    embedding = tf.reduce_mean(embeddings, axis=0)

    prediction = classifier.predict(np.expand_dims(embedding.numpy(), axis=0), verbose=0)
    label = 1 if prediction[0][0] > 0.5 else 0
    timeline.append(label)

    if label == 1:
        snoring_seconds += 1

# Final snoring duration
print(f"\n⏱️ Total snoring time: {snoring_seconds} seconds out of {total_seconds}")

# Visualization
sns.set_style("whitegrid")
plt.figure(figsize=(15, 3))
plt.title('Snoring Activity Timeline')
plt.xlabel('Time (seconds)')
plt.ylabel('Snoring Prediction')

timeline_fuzzy = np.convolve(timeline, np.ones(3)/3, mode='same')

plt.plot(timeline_fuzzy, color='purple', linewidth=1.5, label='Smoothed Snoring Signal')

plt.fill_between(range(len(timeline_fuzzy)), 0, timeline_fuzzy, where=np.array(timeline_fuzzy) > 0.5, 
                 color='red', alpha=0.4, label='Snoring')
plt.fill_between(range(len(timeline_fuzzy)), 0, timeline_fuzzy, where=np.array(timeline_fuzzy) <= 0.5, 
                 color='blue', alpha=0.3, label='Non-snoring')

plt.ylim(0, 1.05)
plt.yticks([0, 0.5, 1], ['Non-snoring', 'Threshold', 'Snoring'])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

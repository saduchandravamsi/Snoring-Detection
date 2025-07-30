import tensorflow as tf
import kagglehub
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# ========== STEP 1: Load Models ==========
print("📦 Downloading and loading YAMNet model...")
yamnet_path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")
yamnet_model = tf.saved_model.load(yamnet_path)
yamnet_infer = yamnet_model.signatures['serving_default']
print("✅ YAMNet model loaded from:", yamnet_path)

print("📦 Loading snoring classifier...")
classifier = tf.keras.models.load_model("snoring_classifier.h5")
print("✅ Snoring classifier model loaded.")

# ========== STEP 2: Load Audio ==========
audio_path = 'm0.wav'  # Replace with your audio
waveform, sr = librosa.load(audio_path, sr=16000)
total_seconds = int(librosa.get_duration(y=waveform, sr=sr))
print(f"🎧 Audio loaded: {total_seconds} seconds")

# ========== STEP 3: Snoring Detection ==========
chunk_samples = sr  # 1 second = 16000 samples
timeline = []
snoring_seconds = 0

print("🔍 Running snoring detection...")
for sec in range(total_seconds):
    start = sec * chunk_samples
    end = start + chunk_samples
    chunk = waveform[start:end]

    if len(chunk) < chunk_samples:
        break

    chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
    results = yamnet_infer(waveform=chunk_tensor)
    embeddings = results['output_1']  # 1024-dim embeddings
    embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()

    prediction = classifier.predict(np.expand_dims(embedding_mean, axis=0), verbose=0)
    label = 1 if prediction[0][0] > 0.5 else 0
    timeline.append(label)
    if label == 1:
        snoring_seconds += 1

print(f"✅ Total snoring detected: {snoring_seconds} seconds")

# ========== STEP 4: OSA Pattern Detection ==========
osa_events = []
pause_count = 0
start_idx = None

for i, label in enumerate(timeline):
    if label == 0:  # Silence or non-snoring
        if start_idx is None:
            start_idx = i
        pause_count += 1
    else:
        if pause_count >= 10:  # Apnea-like pause ≥10s
            osa_events.append((start_idx, i))
        pause_count = 0
        start_idx = None

# ========== STEP 5: OSA Summary ==========
print("\n🩺 OSA-Like Event Summary")
print(f"📌 Audio Duration: {len(timeline)} seconds")
print(f"📌 Total Snoring Seconds: {snoring_seconds}")
print(f"📌 Number of Apnea-Like Events: {len(osa_events)}")
for idx, (start, end) in enumerate(osa_events):
    print(f"  🔴 Event {idx+1}: {start}s to {end}s → {end - start}s pause")

ahi_estimate = (len(osa_events) / (len(timeline) / 60)) * 60
print(f"\n📊 Estimated AHI: {ahi_estimate:.2f} events/hour")

# ========== STEP 6: Visualization ==========
sns.set_style("whitegrid")
plt.figure(figsize=(15, 4))
plt.title("Snoring & OSA Pattern Timeline")
plt.xlabel("Time (seconds)")
plt.ylabel("Prediction")

# Fuzzy smoothing
timeline_fuzzy = np.convolve(timeline, np.ones(3) / 3, mode='same')

# Color code: apnea = orange, snoring = red, non-snoring = blue
colors = []
for i in range(len(timeline)):
    in_apnea = any(start <= i < end for start, end in osa_events)
    if in_apnea:
        colors.append("orange")
    else:
        colors.append("red" if timeline[i] == 1 else "blue")

plt.scatter(range(len(timeline)), timeline_fuzzy, c=colors, s=10)
plt.yticks([0, 0.5, 1], ['Non-snoring', 'Threshold', 'Snoring'])
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

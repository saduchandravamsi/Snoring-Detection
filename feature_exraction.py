import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
from scipy import signal
from tqdm import tqdm

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print(" YAMNet model loaded!")

# Paths
snore_dir = "data/snoring"
non_snore_dir = "data/non_snoring"

X = []
y = []

# Helper function
def load_audio(file_path):
    waveform, sr = sf.read(file_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        waveform = signal.resample_poly(waveform, 16000, sr)
    return waveform

# Process each file
def extract_from_folder(folder, label):
    for file in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        if file.endswith(".wav"):
            try:
                audio_path = os.path.join(folder, file)
                waveform = load_audio(audio_path)
                waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
                _, embeddings, _ = yamnet_model(waveform)
                embedding_mean = tf.reduce_mean(embeddings, axis=0).numpy()
                X.append(embedding_mean)
                y.append(label)
            except Exception as e:
                print(f" Error with {file}: {e}")

# Run on both folders
extract_from_folder(snore_dir, 1)       # Label 1 = snoring
extract_from_folder(non_snore_dir, 0)   # Label 0 = non-snoring

# Save features
X = np.array(X)
y = np.array(y)
np.save("features.npy", X)
np.save("labels.npy", y)

print(" Feature extraction complete. Saved features.npy and labels.npy")

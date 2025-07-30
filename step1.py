import tensorflow_hub as hub
import tensorflow as tf

# Step 1: Load YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Step 2: Print model info
print("YAMNet model loaded!")
print("Model components:", yamnet_model.signatures.keys())

import tensorflow as tf
import kagglehub
path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")
# Load the model from the path
yamnet_model = tf.saved_model.load(path)

# Access the prediction function
yamnet_infer = yamnet_model.signatures['serving_default']

print("✅ YAMNet model loaded from:", path)

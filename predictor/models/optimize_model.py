import tensorflow as tf

# Load the original model
model_path = "parkinson_model.h5"  # Original model file
model = tf.keras.models.load_model(model_path)

# Convert the model to TensorFlow Lite and quantize it
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
optimized_model_path = "parkinson_model_quantized.tflite"  # New optimized model
with open(optimized_model_path, "wb") as f:
    f.write(tflite_model)

print("Model optimized and saved as:", optimized_model_path)
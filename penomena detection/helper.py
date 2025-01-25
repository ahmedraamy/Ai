import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('Model.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)

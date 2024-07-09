# test_and_evaluate.py
import os
import tensorflow as tf
from data_preparation import char_to_num, num_to_char, load_data
import gdown

# Download and extract checkpoints
url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
output = 'checkpoints.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('checkpoints.zip', 'models')

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Load the saved data splits
test = tf.data.Dataset.load('test_data')

# Testing on sample data from the test set
test_data = test.as_numpy_iterator()
sample = test_data.next()
yhat = model.predict(sample[0])

print('~' * 100, 'REAL TEXT')
real_text = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in sample[1]]
for text in real_text:
    print(text.numpy().decode('utf-8'))

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75, 75], greedy=True)[0][0].numpy()

print('~' * 100, 'PREDICTIONS')
predictions = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
for prediction in predictions:
    print(prediction.numpy().decode('utf-8'))

# Testing on a real video
sample = load_data(tf.convert_to_tensor('./data/s1/bras9a.mpg'))

print('~' * 100, 'REAL TEXT')
real_text = tf.strings.reduce_join([num_to_char(word) for word in sample[1]])
print(real_text.numpy().decode('utf-8'))

yhat = model.predict(tf.expand_dims(sample[0], axis=0))
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print('~' * 100, 'PREDICTIONS')
predictions = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
for prediction in predictions:
    print(prediction.numpy().decode('utf-8'))

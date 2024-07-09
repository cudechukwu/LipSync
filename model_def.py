# model_definition.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Activation, MaxPooling3D, TimeDistributed, Flatten, Dense, Reshape, Bidirectional, LSTM, Dropout
from tensorflow.keras.initializers import Orthogonal

# Define the model
model = Sequential()

# Adding Conv3D layers with activation and max pooling
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D((1, 2, 2)))

# Adding TimeDistributed layer to flatten the output of the Conv3D layers
model.add(TimeDistributed(Flatten()))

# Adjusting the input shape for the Dense layer in TimeDistributed
model.add(TimeDistributed(Dense(64)))

# Reshape layer to adjust the shape before feeding into LSTM layers
model.add(Reshape((75, -1)))

# Adding Bidirectional LSTM layers with dropout
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
model.add(Dropout(0.5))

# Final dense layer with softmax activation
model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Save the model architecture for later use
model.save('model.h5')

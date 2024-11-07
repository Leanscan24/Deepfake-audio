from glob import glob

import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     BatchNormalization, Activation, Dropout,
                                     Input)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Define paths to real and fake audio files
real_audio_path = "/mnt/c/Users/cheta/Desktop/Projects/DFAD/dataset/real"
fake_audio_path = "/mnt/c/Users/cheta/Desktop/Projects/DFAD/dataset/fake"

real_audio = glob(real_audio_path + "/*")  # Get list of files in real_audio_path
fake_audio = glob(fake_audio_path + "/*")  # Get list of files in fake_audio_path

real_audio.sort()
fake_audio.sort()

# Print the paths to the real and fake audio files
for i, path in enumerate(real_audio):
    print(i, path.replace(real_audio_path, ""))

print("\n\n")
for i, path in enumerate(fake_audio):
    print(i, path.replace(fake_audio_path, ""))

# Define a function to get the mel spectrogram
def get_mel_spectrogram(y, sr=22050, n_mels=128):
    ms = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    m = lb.power_to_db(ms, ref=np.max)
    return m, m.shape

# Define a function to create mel spectrogram samples
def create_mel_spectrogram_samples(y, label, sr=22050, samples=10, sample_time=3, n_mels=128):
    import random
    import time

    begin = time.time()
    sample_length = int(sr * sample_time)
    s, l = [], []
    for _ in range(samples):
        start = random.randint(0, len(y) - sample_length)
        end = start + sample_length
        m, _ = get_mel_spectrogram(y[start: end], n_mels=n_mels)
        m = np.abs(m)
        m /= 80
        s.append(m)
        l.append([label])

    end = time.time()
    print(
        f"...Sample created with label = '{label}' with {samples} samples | Dimension of mel spectrograms = {m.shape} | Time taken: {end-begin: .3f}s..."
    )
    return np.array(s), np.array(l)

# Create training data
import random

combined_samples = []
combined_labels = []

for audio in real_audio:
    p, sr = lb.load(audio)
    s, l = create_mel_spectrogram_samples(y=p, label=0, samples=200, sample_time=1.5, n_mels=64)
    combined_samples.append(s)
    combined_labels.append(l)

# Check if fake_audio list is empty before proceeding
if fake_audio:
    for audio in random.choices(fake_audio, k=min(8, len(fake_audio))):  # Limit k to the size of fake_audio
        p, sr = lb.load(audio)
        s, l = create_mel_spectrogram_samples(y=p, label=1, samples=200, sample_time=1.5, n_mels=64)
        combined_samples.append(s)
        combined_labels.append(l)
else:
    print("No fake audio files found. Skipping fake audio samples.")

# Check if combined_samples is empty
if combined_samples:
    combined_samples = np.concatenate(combined_samples, axis=0)
    combined_labels = np.concatenate(combined_labels, axis=0)
else:
    print("No audio samples found. Exiting.")
    exit()

# Split data into training, validation, and test sets
print(f"Number of total samples: {combined_samples.shape[0]}")
print(f"Shapes of samples data and labels data: {combined_samples.shape} | {combined_labels.shape}\n")

X, X_test, y, y_test = train_test_split(combined_samples, combined_labels, test_size=0.1, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

print(f"Shapes of train, validation and test data: {X_train.shape} | {X_val.shape} | {X_test.shape}")

# Define the CNN model
model = Sequential([
    Input((64, 65, 1)),

    Conv2D(32, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.20),

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256),
    Activation('relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

# Define callbacks
model_checkpoint = ModelCheckpoint('Model/best_model.keras', monitor='val_accuracy', save_best_only=True, mode="max")
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode="max")

# Train the model
history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Plot the training and validation accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(16, 6))
plt.plot(train_acc, label="train accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
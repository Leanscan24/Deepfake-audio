import streamlit as st
import librosa as lb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile

# Load the pre-trained model
model = tf.keras.models.load_model("Model/best_model.keras")

# Define a function to create mel spectrogram samples
def create_mel_spectrogram_samples(y, sr=22050, samples=10, sample_time=3, n_mels=128):
    import random
    import time

    begin = time.time()
    sample_length = int(sr * sample_time)
    s = []
    for _ in range(samples):
        start = random.randint(0, len(y) - sample_length)
        end = start + sample_length
        m, _ = get_mel_spectrogram(y[start: end], n_mels=n_mels)
        m = np.abs(m)
        m /= 80
        s.append(m)

    end = time.time()
    print(
        f"...Sample created with {samples} samples | Dimension of mel spectrograms = {m.shape} | Time taken: {end-begin: .3f}s..."
    )
    return np.array(s)

# Define a function to get the mel spectrogram
def get_mel_spectrogram(y, sr=22050, n_mels=128):
    ms = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    m = lb.power_to_db(ms, ref=np.max)
    return m, m.shape

# Define the Streamlit app
def app():
    # Set the title of the app
    st.title("DeepFake Voice Detector")

    # Add a file uploader to the app
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    # If a file is uploaded
    if uploaded_file is not None:
         # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the audio file
        y, sr = lb.load(tmp_file_path, sr=22050)

        # Create mel spectrogram samples
        samples = create_mel_spectrogram_samples(
            y, sr=sr, samples=10, sample_time=1.5, n_mels=64
        )

        # Make predictions
        predictions = model.predict(samples)
        average_prediction = np.mean(predictions)

        # Display the results
        st.write("### Predictions:")
        if average_prediction > 0.5:
            st.error("The audio file is likely to be a fake voice.")
        else:
            st.success("The audio file is likely to be a real voice.")

        # Visualize the mel spectrogram
        st.write("### Mel Spectrogram:")
        m, _ = get_mel_spectrogram(y, sr=sr, n_mels=64)
        fig, ax = plt.subplots(figsize=(8, 4))
        img = lb.display.specshow(m, x_axis="time", y_axis="mel", sr=sr, ax=ax)
        plt.colorbar(img, format="%2.0f dB")
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    app()
from settings import *
from tetris import Tetris, Text
import sys
import pathlib
import sounddevice as sd
import soundfile as sf
from python_speech_features import mfcc
from sklearn.preprocessing import normalize, StandardScaler
from joblib import load
import os
import numpy as np
from scipy.signal import lfilter, butter
import pickle
from whisper import load_model,transcribe

class App:
    def __init__(self):
        pg.init()
        pg.display.set_caption('Tetris')
        self.screen = pg.display.set_mode(WIN_RES)
        self.clock = pg.time.Clock()
        self.set_timer()
        self.images = self.load_images()
        self.tetris = Tetris(self)
        self.text = Text(self)
        self.recording_audio = False

    def load_images(self):
        files = [item for item in pathlib.Path(SPRITE_DIR_PATH).rglob('*.png') if item.is_file()]
        images = [pg.image.load(file).convert_alpha() for file in files]
        images = [pg.transform.scale(image, (TILE_SIZE, TILE_SIZE)) for image in images]
        return images

    def set_timer(self):
        self.user_event = pg.USEREVENT + 0
        self.fast_user_event = pg.USEREVENT + 1
        self.anim_trigger = False
        self.fast_anim_trigger = False
        pg.time.set_timer(self.user_event, ANIM_TIME_INTERVAL)
        pg.time.set_timer(self.fast_user_event, FAST_ANIM_TIME_INTERVAL)

    def update(self):
        self.tetris.update()
        self.clock.tick(FPS)

    def draw(self):
        self.screen.fill(color=BG_COLOR)
        self.screen.fill(color=FIELD_COLOR, rect=(0, 0, *FIELD_RES))
        self.tetris.draw()
        self.text.draw()
        pg.display.flip()
    
    def record_audio(self):
        file_path = "Recordings"
        noise_reduction=True
        filter_length=100
        volume_scale=20
        sr = 16000
        duration = 1.2
        print("Listening")
        audio = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        # Apply noise reduction
        if noise_reduction:
            cutoff_freq = 300.0  # Hz
            b, a = butter(5, cutoff_freq / (sr / 2), btype='high')
            audio = lfilter(b, a, audio)

            # Apply moving average filter for further noise reduction
            audio = lfilter(np.ones(filter_length) / filter_length, 1, audio)

        audio *= volume_scale

        # Normalize audio to prevent clipping
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.6:
            audio /= max_amplitude

        # Save audio to WAV file
        file_path = os.path.join(file_path, "test.wav")
        sf.write(file_path, audio, samplerate=sr, subtype='PCM_16')
        self.predict_label(file_path)


    def predict_label(self, file_path):
        audio, sr = sf.read(file_path)

        # Normalize audio
        normalized_audio = normalize(audio.reshape(1, -1)).reshape(-1)

        # Extract MFCC features
        mfcc_features = mfcc(normalized_audio, sr, nfft=2048)

        # Normalize the features
        scaler = StandardScaler()
        mfcc_features_scaled = scaler.fit_transform(mfcc_features)

        # Reshape the data for compatibility with classifiers
        mfcc_features_flat = mfcc_features_scaled.reshape(1, -1)

        # Limit the features to 1287
        mfcc_features_1287 = mfcc_features_flat[:, :1287]

        model = load("Models/SVM_model.joblib")
        # Make a prediction using the loaded model
        predicted_label = model.predict(mfcc_features_1287.reshape(1, -1))

        label_encoder = load("Models/label_encoder.joblib")
        # Decode the predicted label
        decoded_label = label_encoder.inverse_transform(predicted_label)
        print(f"Predicted label: {decoded_label}")
        self.tetris.handle_voice_command(command=decoded_label)

    def speech_recognition(self):
        file_path = "Recordings"
        noise_reduction=True
        filter_length=100
        volume_scale=20
        sr = 16000
        duration = 1.2
        print("Listening")
        audio = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        # Apply noise reduction
        if noise_reduction:
            cutoff_freq = 300.0  # Hz
            b, a = butter(5, cutoff_freq / (sr / 2), btype='high')
            audio = lfilter(b, a, audio)

            # Apply moving average filter for further noise reduction
            audio = lfilter(np.ones(filter_length) / filter_length, 1, audio)

        audio *= volume_scale

        # Normalize audio to prevent clipping
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.6:
            audio /= max_amplitude

        # Save audio to WAV file
        file_path = os.path.join(file_path, "test.wav")
        sf.write(file_path, audio, samplerate=sr, subtype='PCM_16')
        with open("Models/base_en.pkl", "rb") as f:
            model = pickle.load(f)
        result = model.transcribe(file_path)
        command = result["text"].lower()
        print("Predicted label:" + command)
        self.tetris.handle_voice_command(command=command)
    
    def check_events(self):
        self.anim_trigger = False
        self.fast_anim_trigger = False
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                self.tetris.control(pressed_key=event.key)
            elif event.type == self.user_event:
                self.anim_trigger = True
            elif event.type == self.fast_user_event:
                self.fast_anim_trigger = True
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self.record_audio()
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
                self.speech_recognition()

    def run(self):
        while True:
            self.check_events()
            self.update()
            self.draw()


if __name__ == '__main__':
    app = App()
    app.run()

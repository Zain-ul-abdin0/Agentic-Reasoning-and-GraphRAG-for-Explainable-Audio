import librosa
import numpy as np
import parselmouth


class AudioFeatureExtractor:

    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(
            audio_path,
            sr=16000
        )

    def extract_mfcc(self):

        mfcc = librosa.feature.mfcc(
            y=self.audio,
            sr=self.sr,
            n_mfcc=13
        )

        return np.mean(mfcc, axis=1)

    def extract_energy(self):

        rms = librosa.feature.rms(
            y=self.audio
        )

        return float(np.mean(rms))

    def extract_pitch_variability(self):

        pitches, magnitudes = librosa.piptrack(
            y=self.audio,
            sr=self.sr
        )

        pitch_values = []

        for i in range(pitches.shape[1]):

            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]

            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) == 0:
            return 0

        return float(np.std(pitch_values))

    def extract_pause_ratio(self):

        intervals = librosa.effects.split(
            self.audio,
            top_db=20
        )

        speech_samples = 0

        for start, end in intervals:
            speech_samples += (end - start)

        speech_duration = speech_samples / self.sr
        total_duration = len(self.audio) / self.sr

        pause_duration = total_duration - speech_duration

        return float(
            pause_duration / total_duration
        )

    def extract_jitter(self):

        sound = parselmouth.Sound(
            self.audio_path
        )

        point_process = parselmouth.praat.call(
            sound,
            "To PointProcess (periodic, cc)",
            75,
            500
        )

        jitter = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3
        )

        return float(jitter)

    def extract_all(self):

        return {

            "energy":
                self.extract_energy(),

            "pitch_std":
                self.extract_pitch_variability(),

            "pause_ratio":
                self.extract_pause_ratio(),

            "jitter":
                self.extract_jitter(),

            "mfcc":
                self.extract_mfcc().tolist()
        }
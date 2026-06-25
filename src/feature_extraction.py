import librosa
import numpy as np
import parselmouth


class AudioFeatureExtractor:

    def __init__(self, audio_path, sample_rate=16000):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.audio, self.sr = librosa.load(
            audio_path,
            sr=sample_rate,
            mono=True
        )

    def extract_duration(self):
        return float(librosa.get_duration(y=self.audio, sr=self.sr))

    def extract_mfcc(self, n_mfcc=13):

        mfcc = librosa.feature.mfcc(
            y=self.audio,
            sr=self.sr,
            n_mfcc=n_mfcc
        )

        return {
            "mean": np.mean(mfcc, axis=1),
            "std": np.std(mfcc, axis=1)
        }

    def extract_energy(self):

        rms = librosa.feature.rms(
            y=self.audio
        )

        return float(np.mean(rms))

    def extract_pitch(self):

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
            return {
                "mean": 0.0,
                "std": 0.0
            }

        return {
            "mean": float(np.mean(pitch_values)),
            "std": float(np.std(pitch_values))
        }

    def extract_pause_ratio(self):

        total_duration = self.extract_duration()
        if total_duration == 0:
            return 0.0

        intervals = librosa.effects.split(
            self.audio,
            top_db=20
        )

        speech_samples = 0

        for start, end in intervals:
            speech_samples += (end - start)

        speech_duration = speech_samples / self.sr

        pause_duration = total_duration - speech_duration

        return float(
            pause_duration / total_duration
        )

    def extract_jitter(self):
        try:
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

            if np.isnan(jitter):
                return 0.0

            return float(jitter)
        except Exception:
            return 0.0

    def extract_all(self):
        pitch = self.extract_pitch()
        mfcc = self.extract_mfcc()

        return {
            "duration_seconds":
                self.extract_duration(),

            "energy":
                self.extract_energy(),

            "pitch_mean":
                pitch["mean"],

            "pitch_std":
                pitch["std"],

            "pause_ratio":
                self.extract_pause_ratio(),

            "jitter":
                self.extract_jitter(),

            "mfcc_mean":
                mfcc["mean"].tolist(),

            "mfcc_std":
                mfcc["std"].tolist()
        }

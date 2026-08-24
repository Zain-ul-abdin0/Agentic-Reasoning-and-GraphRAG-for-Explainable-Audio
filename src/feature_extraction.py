from pathlib import Path
import csv
import math
import wave

import numpy as np
import parselmouth
from scipy.fftpack import dct


def _parse_float(value):
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except ValueError:
        return None


def _read_csv_rows(path):
    path = Path(path)
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:2048]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","

    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def _tokenize(text):
    return [
        token
        for token in text.replace("'", " ").split()
        if token.strip()
    ]


def extract_transcript_features(transcript_path, max_duration_seconds=None):
    participant_turns = []

    for row in _read_csv_rows(transcript_path):
        if row.get("speaker") != "Participant":
            continue

        start = _parse_float(row.get("start_time"))
        stop = _parse_float(row.get("stop_time"))
        if start is None or stop is None or stop <= start:
            continue

        if max_duration_seconds is not None:
            if start >= max_duration_seconds:
                continue
            stop = min(stop, max_duration_seconds)
            if stop <= start:
                continue

        participant_turns.append({
            "start": start,
            "stop": stop,
            "words": len(_tokenize(row.get("value", ""))),
        })

    if not participant_turns:
        return {}

    speech_duration = sum(
        turn["stop"] - turn["start"]
        for turn in participant_turns
    )
    word_count = sum(turn["words"] for turn in participant_turns)
    start_time = min(turn["start"] for turn in participant_turns)
    stop_time = max(turn["stop"] for turn in participant_turns)
    total_span = max(stop_time - start_time, speech_duration)
    pause_duration = max(total_span - speech_duration, 0.0)

    return {
        "speech_rate": word_count / speech_duration if speech_duration else None,
        "pause_ratio": pause_duration / total_span if total_span else None,
        "participant_speech_duration": speech_duration,
        "participant_word_count": word_count,
    }


def _stream_column_values(path, column_index, positive_only=False):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) <= column_index:
                continue

            value = _parse_float(row[column_index])
            if value is None:
                continue
            if positive_only and value <= 0:
                continue

            yield value


def extract_covarep_features(covarep_path):
    f0_values = list(_stream_column_values(covarep_path, 0, positive_only=True))
    if len(f0_values) < 2:
        return {}

    f0 = np.asarray(f0_values, dtype=np.float64)
    periods = 1.0 / f0
    period_diffs = np.abs(np.diff(periods))
    jitter_proxy = float(np.mean(period_diffs) / np.mean(periods))

    return {
        "pitch_mean": float(np.mean(f0)),
        "pitch_std": float(np.std(f0)),
        "jitter": jitter_proxy,
    }


class AudioFeatureExtractor:

    def __init__(
        self,
        audio_path,
        sample_rate=16000,
        max_duration_seconds=None,
        transcript_path=None,
        covarep_path=None
    ):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.transcript_path = transcript_path
        self.covarep_path = covarep_path
        self.audio, self.sr = self.read_wav_mono(audio_path)
        self._praat_sound = None

    def read_wav_mono(self, audio_path):
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            source_rate = wav.getframerate()
            frame_count = wav.getnframes()

            if self.max_duration_seconds is not None:
                frame_count = min(
                    frame_count,
                    int(source_rate * self.max_duration_seconds)
                )

            frames = wav.readframes(frame_count)

        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            audio = audio / 2147483648.0
        else:
            raise ValueError(
                f"Unsupported WAV sample width {sample_width}; expected 16-bit or 32-bit PCM."
            )

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if source_rate != self.sample_rate and audio.size:
            audio = self.resample_linear(audio, source_rate, self.sample_rate)

        return audio.astype(np.float32), self.sample_rate

    @staticmethod
    def resample_linear(audio, source_rate, target_rate):
        duration = audio.size / source_rate
        target_size = max(int(duration * target_rate), 1)
        old_positions = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        new_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
        return np.interp(new_positions, old_positions, audio).astype(np.float32)

    def praat_sound(self):
        if self._praat_sound is None:
            self._praat_sound = parselmouth.Sound(
                self.audio,
                sampling_frequency=self.sr
            )

        return self._praat_sound

    def extract_duration(self):
        return float(self.audio.size / self.sr) if self.sr else 0.0

    @staticmethod
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def mel_filterbank(self, n_fft, n_mels=26):
        low_mel = self.hz_to_mel(0)
        high_mel = self.hz_to_mel(self.sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for index in range(1, n_mels + 1):
            left = bins[index - 1]
            center = bins[index]
            right = bins[index + 1]

            if center > left:
                filters[index - 1, left:center] = (
                    np.arange(left, center) - left
                ) / (center - left)
            if right > center:
                filters[index - 1, center:right] = (
                    right - np.arange(center, right)
                ) / (right - center)

        return filters

    def frame_audio(self, frame_ms=25, hop_ms=10, max_frames=5000):
        frame_length = int(self.sr * frame_ms / 1000)
        hop_length = int(self.sr * hop_ms / 1000)

        if self.audio.size < frame_length:
            return np.empty((0, frame_length), dtype=np.float32)

        frame_count = 1 + (self.audio.size - frame_length) // hop_length
        starts = np.arange(frame_count) * hop_length

        if frame_count > max_frames:
            starts = starts[
                np.linspace(0, frame_count - 1, max_frames).astype(int)
            ]

        return np.stack([
            self.audio[start:start + frame_length]
            for start in starts
        ]).astype(np.float32)

    def extract_mfcc(self, n_mfcc=13):
        frames = self.frame_audio()

        if frames.size == 0:
            zeros = np.zeros(n_mfcc, dtype=np.float32)
            return {
                "mean": zeros,
                "std": zeros
            }

        emphasized = np.empty_like(frames)
        emphasized[:, 0] = frames[:, 0]
        emphasized[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        windowed = emphasized * np.hamming(emphasized.shape[1])

        n_fft = 512
        spectrum = np.fft.rfft(windowed, n=n_fft)
        power = (np.abs(spectrum) ** 2) / n_fft
        mel_energy = np.dot(power, self.mel_filterbank(n_fft).T)
        log_mel = np.log(np.maximum(mel_energy, 1e-10))
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

        return {
            "mean": np.mean(mfcc, axis=0),
            "std": np.std(mfcc, axis=0)
        }

    def extract_energy(self):
        if self.audio.size == 0:
            return 0.0

        return float(np.sqrt(np.mean(np.square(self.audio))))

    def extract_pitch(self):
        pitch = self.praat_sound().to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500
        )
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]

        if pitch_values.size == 0:
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

        frames = self.frame_audio()
        if frames.size == 0:
            return 1.0

        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        threshold = max(float(np.max(frame_rms)) * 0.1, 1e-6)
        speech_frames = int(np.sum(frame_rms > threshold))
        speech_ratio = speech_frames / len(frame_rms)

        return float(1.0 - speech_ratio)

    def extract_jitter(self):
        try:
            point_process = parselmouth.praat.call(
                self.praat_sound(),
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
        covarep_features = {}
        if self.covarep_path and Path(self.covarep_path).exists():
            covarep_features = extract_covarep_features(self.covarep_path)

        if covarep_features:
            pitch = {
                "mean": covarep_features.get("pitch_mean", 0.0),
                "std": covarep_features.get("pitch_std", 0.0),
            }
            jitter = covarep_features.get("jitter", 0.0)
        else:
            pitch = self.extract_pitch()
            jitter = self.extract_jitter()

        mfcc = self.extract_mfcc()
        features = {
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
                jitter,

            "mfcc_mean":
                mfcc["mean"].tolist(),

            "mfcc_std":
                mfcc["std"].tolist()
        }
        feature_sources = {
            "duration_seconds": "audio",
            "energy": "audio",
            "pitch_mean": "covarep" if covarep_features else "audio_praat",
            "pitch_std": "covarep" if covarep_features else "audio_praat",
            "pause_ratio": "audio_energy_frames",
            "jitter": "covarep" if covarep_features else "audio_praat",
            "mfcc_mean": "audio_mfcc",
            "mfcc_std": "audio_mfcc",
        }

        if self.transcript_path and Path(self.transcript_path).exists():
            transcript_features = extract_transcript_features(
                self.transcript_path,
                max_duration_seconds=self.max_duration_seconds
            )
            for key, value in transcript_features.items():
                if value is not None:
                    features[key] = value
                    feature_sources[key] = "transcript"

        features["feature_sources"] = feature_sources
        return features

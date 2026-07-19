from pathlib import Path
import argparse
import csv
import math
import wave

import numpy as np
from scipy.fftpack import dct

from src.calibration.calibrate_acoustic_thresholds import (
    DEFAULT_LABEL_FILES,
    extract_transcript_features,
    load_phq_labels,
    parse_float,
    resolve_path,
    stream_column_values,
)


def read_wav_mono(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}: {path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def rms_energy(audio):
    if audio.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(audio))))


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate, n_fft, n_mels=26):
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

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


def frame_audio(audio, sample_rate, frame_ms=25, hop_ms=10, max_frames=5000):
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    if audio.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)

    frame_count = 1 + (audio.size - frame_length) // hop_length
    starts = np.arange(frame_count) * hop_length

    if frame_count > max_frames:
        starts = starts[np.linspace(0, frame_count - 1, max_frames).astype(int)]

    frames = np.stack([
        audio[start:start + frame_length]
        for start in starts
    ]).astype(np.float32)
    return frames


def mfcc_features(audio, sample_rate, n_mfcc=13):
    frames = frame_audio(audio, sample_rate)
    if frames.size == 0:
        return {}

    emphasized = np.empty_like(frames)
    emphasized[:, 0] = frames[:, 0]
    emphasized[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    windowed = emphasized * np.hamming(emphasized.shape[1])

    n_fft = 512
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft
    mel_energy = np.dot(power, mel_filterbank(sample_rate, n_fft).T)
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    coefficients = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    result = {}
    for index in range(min(4, n_mfcc)):
        result[f"mfcc_mean_{index}"] = float(np.mean(coefficients[:, index]))
        result[f"mfcc_std_{index}"] = float(np.std(coefficients[:, index]))

    return result


def covarep_pitch_and_jitter(dataset_dir, participant_id):
    path = resolve_path(dataset_dir) / f"{participant_id}_COVAREP.csv"
    if not path.exists():
        return {}

    f0_values = list(stream_column_values(path, 0, positive_only=True))
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


def write_matrix(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({
        key
        for row in rows
        for key in row
        if key not in {"participant_id", "phq_score", "phq8_binary"}
    })
    fieldnames = [
        "participant_id",
        "phq_score",
        "phq8_binary",
        *feature_names,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(dataset_dir, labels, limit=None):
    dataset_path = resolve_path(dataset_dir)
    transcript_features = extract_transcript_features(dataset_dir)
    rows = []

    participant_ids = sorted(
        participant_id
        for participant_id in labels
        if (dataset_path / f"{participant_id}_AUDIO.wav").exists()
    )

    if limit is not None:
        participant_ids = participant_ids[:limit]

    for participant_id in participant_ids:
        audio_path = dataset_path / f"{participant_id}_AUDIO.wav"
        audio, sample_rate = read_wav_mono(audio_path)

        row = {
            "participant_id": participant_id,
            "phq_score": labels[participant_id]["phq_score"],
            "phq8_binary": labels[participant_id]["label"],
            "energy": rms_energy(audio),
        }
        row.update(transcript_features.get(participant_id, {}))
        row.update(covarep_pitch_and_jitter(dataset_dir, participant_id))
        row.update(mfcc_features(audio, sample_rate))
        rows.append(row)

        print(f"extracted participant {participant_id}", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build participant-level acoustic features for threshold calibration."
    )
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument(
        "--label-file",
        action="append",
        default=None,
        help="CSV with Participant_ID and PHQ8_Score/PHQ_Score columns. Can be repeated.",
    )
    parser.add_argument("--phq-cutoff", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default="data/month_02_knowledge_graph/calibration/acoustic_feature_matrix.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional participant limit for quick testing.",
    )

    args = parser.parse_args()
    labels = load_phq_labels(args.label_file or DEFAULT_LABEL_FILES, args.phq_cutoff)
    rows = build_matrix(args.dataset_dir, labels, args.limit)
    write_matrix(resolve_path(args.output), rows)

    print(f"wrote {len(rows)} rows to {resolve_path(args.output)}")


if __name__ == "__main__":
    main()

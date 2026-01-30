"""
Human Noise Detection System (Integrated)
Select an audio file using Windows file dialog and detect human voice/speech.
Uses Silero VAD - a pre-trained deep learning model for accurate voice detection.
Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
import torch
import torchaudio
import librosa

# Windows file dialog
import tkinter as tk
from tkinter import filedialog

# Supported audio formats
SUPPORTED_FORMATS = [
    ('Audio Files', '*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;*.wma;*.opus'),
    ('WAV files', '*.wav'),
    ('MP3 files', '*.mp3'),
    ('FLAC files', '*.flac'),
    ('All files', '*.*')
]

# Global model variable (loaded once)
_vad_model = None
_vad_utils = None


def load_silero_vad():
    """
    Load Silero VAD model from torch.hub.
    Model is cached after first download.
    
    Returns:
        Tuple of (model, utils)
    """
    global _vad_model, _vad_utils
    
    if _vad_model is None:
        print("  Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        _vad_model = model
        _vad_utils = utils
        print("  Model loaded successfully!")
    
    return _vad_model, _vad_utils


def calculate_audio_energy(audio_data):
    """
    Calculate the energy/loudness of an audio segment.
    Returns normalized energy level (0-1).
    """
    if len(audio_data) == 0:
        return 0.0
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    return float(rms)


def auto_detect_threshold(audio_data, sr):
    """
    Auto-detect the audio energy threshold based on the audio content.
    """
    chunk_samples = int(0.1 * sr)  # 100ms chunks
    
    if len(audio_data) < chunk_samples * 2:
        return 0.01
    
    energies = []
    for i in range(0, len(audio_data) - chunk_samples, chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        if len(chunk) > 0:
            energies.append(calculate_audio_energy(chunk))
    
    if len(energies) == 0:
        return 0.01
    
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    threshold = max(mean_energy - (std_energy * 0.5), 0.005)
    
    return threshold


def detect_human_voice(audio_data, sr):
    """
    Detect human voice in audio using Silero VAD deep learning model.
    
    Returns:
        Tuple of (is_human_detected, confidence_percentage, high_energy_detected)
    """
    # Load Silero VAD model
    model, utils = load_silero_vad()
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Calculate energy level
    energy = calculate_audio_energy(audio_data)
    threshold = auto_detect_threshold(audio_data, sr)
    high_energy = energy > threshold
    
    # If energy is too low, it's likely silence
    if not high_energy:
        return False, 0.0, False
    
    # Silero VAD requires 16kHz sample rate
    target_sr = 16000
    if sr != target_sr:
        audio_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    else:
        audio_resampled = audio_data
    
    # Normalize audio for better detection
    max_val = np.max(np.abs(audio_resampled))
    if max_val > 0:
        audio_resampled = audio_resampled / max_val
    
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_resampled).float()
    
    # Get speech timestamps using Silero VAD with low threshold
    try:
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            model, 
            sampling_rate=target_sr,
            threshold=0.2,  # Low threshold for better sensitivity
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
        )
    except Exception as e:
        print(f"  Warning: VAD error - {str(e)}")
        return False, 0.0, high_energy
    
    # Calculate speech percentage
    total_samples = len(audio_resampled)
    speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
    
    if total_samples > 0:
        speech_ratio = speech_samples / total_samples
        confidence = speech_ratio * 100
    else:
        confidence = 0.0
    
    # Check if it looks like a siren (oscillating pure tone)
    is_siren_like = False
    if confidence < 20.0:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_resampled, sr=target_sr)[0]
        centroid_std = np.std(spectral_centroid)
        centroid_mean = np.mean(spectral_centroid)
        
        if centroid_mean > 400 and centroid_std > 100:
            centroid_diff = np.diff(spectral_centroid)
            smoothness = np.mean(np.abs(np.diff(centroid_diff)))
            if smoothness < 50:
                is_siren_like = True
    
    # Primary detection: Silero VAD found speech AND not siren-like
    if len(speech_timestamps) > 0 and speech_ratio > 0.01 and not is_siren_like:
        return True, confidence, high_energy
    
    # Check for voice characteristics if VAD missed but energy is high
    if high_energy and not is_siren_like and speech_ratio > 0.005:
        return True, max(confidence, 5.0), high_energy
    
    return False, confidence, high_energy


def load_audio_file(file_path):
    """
    Load audio file in any supported format using librosa.
    
    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) if failed
    """
    try:
        # First try with librosa (handles most formats)
        audio_data, sr = librosa.load(file_path, sr=16000, mono=True)
        return audio_data, sr
    except Exception as e:
        # Try with torchaudio as fallback
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                sr = 16000
            audio_data = waveform.squeeze().numpy()
            return audio_data, sr
        except Exception as e2:
            print(f"  Error loading audio: {str(e2)}")
            return None, None


def format_duration(seconds):
    """Format duration in a human-readable way."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def select_audio_file():
    """
    Open a Windows file dialog to select an audio file.
    
    Returns:
        Selected file path or None if cancelled
    """
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Audio File for Human Detection",
        filetypes=SUPPORTED_FORMATS,
        initialdir=os.path.expanduser("~")
    )
    
    root.destroy()
    
    return file_path if file_path else None


def analyze_audio(file_path):
    """
    Analyze a single audio file for human presence.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary with detection results
    """
    filename = os.path.basename(file_path)
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {filename}")
    print('='*60)
    
    # Load audio
    audio_data, sr = load_audio_file(file_path)
    if audio_data is None:
        print("  ERROR: Could not load audio file!")
        return {"success": False, "error": "Could not load audio file"}
    
    # Print audio info
    duration = format_duration(len(audio_data) / sr)
    print(f"  Duration: {duration}")
    print(f"  Sample Rate: {sr} Hz")
    
    # Detect human voice
    print("\n  Analyzing for human voice...")
    is_human, confidence, high_energy = detect_human_voice(audio_data, sr)
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"  Energy Level: {'HIGH' if high_energy else 'LOW'}")
    print(f"  Voice Confidence: {confidence:.1f}%")
    
    if is_human:
        print("\n  ┌─────────────────────────────────────┐")
        print("  │  ✅ HUMAN PRESENCE DETECTED         │")
        print("  └─────────────────────────────────────┘")
    else:
        print("\n  ┌─────────────────────────────────────┐")
        print("  │  ❌ NO HUMAN DETECTED               │")
        print("  └─────────────────────────────────────┘")
    
    print('='*60)
    
    return {
        "success": True,
        "file": filename,
        "is_human_present": is_human,
        "confidence": confidence,
        "high_energy": high_energy
    }


def main():
    """Main function - select file and analyze."""
    print("\n" + "="*60)
    print("  HUMAN PRESENCE DETECTOR")
    print("  Using Silero VAD Deep Learning Model")
    print("="*60)
    
    while True:
        print("\nSelect an audio file to analyze...")
        
        # Open file selection dialog
        file_path = select_audio_file()
        
        if not file_path:
            print("\nNo file selected. Exiting.")
            break
        
        # Analyze the selected file
        result = analyze_audio(file_path)
        
        # Ask if user wants to analyze another file
        print("\n" + "-"*60)
        response = input("Analyze another file? (y/n): ").strip().lower()
        if response != 'y':
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

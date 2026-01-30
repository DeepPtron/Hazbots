"""
Human Noise Detection System (Silero VAD)
Scans a folder for audio files and detects human voice/speech.
Uses Silero VAD - a pre-trained deep learning model for accurate voice detection.
Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
import torch
import torchaudio
import librosa

# Add local FFmpeg to PATH if present
ffmpeg_path = r"C:\Users\niran\OneDrive\Desktop\nvenv"
if os.path.exists(os.path.join(ffmpeg_path, "ffmpeg.exe")):
    os.environ["PATH"] += os.pathsep + ffmpeg_path
    print(f"  Found local FFmpeg, added to PATH: {ffmpeg_path}")

# Supported audio formats
SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus']

# Folder path - SET YOUR FOLDER PATH HERE
AUDIO_FOLDER_PATH = r"C:\Users\niran\OneDrive\Desktop\test f"  # Example: "C:/Users/YourName/AudioFiles"

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
    
    Args:
        audio_data: numpy array of audio samples (float32)
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
    
    Args:
        audio_data: numpy array of audio samples
        sr: sample rate of the audio
    
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
    # Only check for very low-confidence detections (below 20%)
    is_siren_like = False
    if confidence < 20.0:
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_resampled, sr=target_sr)[0]
        centroid_std = np.std(spectral_centroid)
        centroid_mean = np.mean(spectral_centroid)
        
        # Sirens have smooth, predictable centroid changes (oscillating pattern)
        # Voice has more erratic centroid changes
        if centroid_mean > 400 and centroid_std > 100:  # Has varying frequency
            # Check if the variation is too smooth (siren-like)
            centroid_diff = np.diff(spectral_centroid)
            smoothness = np.mean(np.abs(np.diff(centroid_diff)))  # Second derivative
            if smoothness < 50:  # Very smooth = siren
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
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) if failed
    """
    try:
        # First try with librosa (handles most formats)
        audio_data, sr = librosa.load(file_path, sr=16000, mono=True)  # Use 16kHz for VAD
        return audio_data, sr
    except Exception as e:
        # Try with torchaudio as fallback for AAC and other formats
        try:
            waveform, sr = torchaudio.load(file_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample to 16kHz for VAD
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                sr = 16000
            audio_data = waveform.squeeze().numpy()
            return audio_data, sr
        except Exception as e2:
            print(f"  Error loading {os.path.basename(file_path)}: {str(e2)}")
            return None, None


def get_audio_files(folder_path):
    """
    Get all audio files from the specified folder.
    
    Args:
        folder_path: Path to the folder containing audio files
    
    Returns:
        List of audio file paths
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return []
    
    audio_files = []
    for filename in os.listdir(folder_path):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            audio_files.append(os.path.join(folder_path, filename))
    
    return sorted(audio_files)


def format_duration(seconds):
    """Format duration in a human-readable way."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def process_audio_folder(folder_path=None):
    """
    Main function to process all audio files in a folder.
    
    Args:
        folder_path: Path to folder with audio files. If None, uses AUDIO_FOLDER_PATH.
    """
    # Use provided path or default
    path = folder_path if folder_path else AUDIO_FOLDER_PATH
    
    if not path or path.strip() == "":
        print("=" * 60)
        print("HUMAN NOISE DETECTION SYSTEM (Silero VAD)")
        print("=" * 60)
        print("\nNo folder path specified!")
        print("   Please set the AUDIO_FOLDER_PATH variable at the top of this script")
        print("   Or call process_audio_folder('your/folder/path')")
        print("\n   Example:")
        print("   AUDIO_FOLDER_PATH = 'C:/Users/YourName/AudioFiles'")
        print("=" * 60)
        return
    
    # Get all audio files
    audio_files = get_audio_files(path)
    
    print("=" * 60)
    print("HUMAN NOISE DETECTION SYSTEM (Silero VAD)")
    print("=" * 60)
    print(f"\nFolder: {path}")
    print(f"Found {len(audio_files)} audio file(s)")
    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    print("=" * 60)
    
    if len(audio_files) == 0:
        print("\nNo audio files found in the specified folder!")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    # Process each file
    detected_count = 0
    detected_files = []  # Track files with voice detected
    
    for i, file_path in enumerate(audio_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(audio_files)}] Processing: {filename}")
        
        # Load audio
        audio_data, sr = load_audio_file(file_path)
        if audio_data is None:
            print("  Skipping (could not load)")
            continue
        
        # Print audio info
        duration = format_duration(len(audio_data) / sr)
        channels = "Mono" if len(audio_data.shape) == 1 else f"{audio_data.shape[1]} channels"
        print(f"  Duration: {duration} | {channels} | Sample Rate: {sr}Hz")
        
        # Detect human voice
        is_human, confidence, high_energy = detect_human_voice(audio_data, sr)
        
        if is_human:
            print(f"  Energy Level: {'HIGH' if high_energy else 'LOW'}")
            print(f"  Voice Confidence: {confidence:.1f}%")
            print("  >> Voice Detected")
            detected_count += 1
            detected_files.append(filename)
        else:
            print(f"  Energy Level: {'HIGH' if high_energy else 'LOW'}")
            print(f"  Voice Confidence: {confidence:.1f}%")
            print("  >> No human noise - moving to next file...")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total files processed: {len(audio_files)}")
    print(f"  Voice detected in: {detected_count} file(s)")
    print(f"  No voice in: {len(audio_files) - detected_count} file(s)")
    
    if detected_files:
        print("\n  Files with voice detected:")
        for f in detected_files:
            print(f"    - {f}")
    print("=" * 60)


# Run the detection when script is executed directly
if __name__ == "__main__":
    process_audio_folder()

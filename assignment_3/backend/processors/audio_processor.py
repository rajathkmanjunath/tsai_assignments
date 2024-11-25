from .base_processor import BaseProcessor
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random
import soundfile as sf
from pathlib import Path

class AudioProcessor(BaseProcessor):
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate
        self.duration = 30  # Max duration in seconds
        
        # Augmentation techniques with mild parameters
        self.augmentation_techniques = [
            self.time_stretch,
            self.pitch_shift,
            self.add_noise,
            self.change_volume
        ]

    def preprocess(self, audio_path):
        """
        Preprocess audio:
        - Load and resample
        - Convert to mono if stereo
        - Trim silence
        - Normalize amplitude
        """
        # Load audio with a consistent sample rate
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio, sr

    def time_stretch(self, audio, sr):
        """Time stretching"""
        rate = random.uniform(0.9, 1.1)  # Mild time stretching
        return librosa.effects.time_stretch(audio, rate=rate), f"Time Stretch (rate={rate:.2f})"

    def pitch_shift(self, audio, sr):
        """Pitch shifting"""
        steps = random.uniform(-2, 2)  # Mild pitch shift
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps), f"Pitch Shift (steps={steps:.2f})"

    def add_noise(self, audio, sr):
        """Add mild noise"""
        noise_level = random.uniform(0.001, 0.002)
        noise = np.random.randn(len(audio))
        return audio + noise * noise_level, f"Add Noise (level={noise_level:.3f})"

    def change_volume(self, audio, sr):
        """Change volume"""
        volume_factor = random.uniform(0.8, 1.2)
        return audio * volume_factor, f"Volume Change (factor={volume_factor:.2f})"

    def augment(self, audio, sr):
        """
        Apply random augmentations:
        - Select 1-2 random augmentation techniques
        - Apply them sequentially
        """
        # Select 1-2 random augmentation techniques
        num_augmentations = random.randint(1, 2)
        selected_techniques = random.sample(self.augmentation_techniques, num_augmentations)
        
        augmented = audio.copy()
        applied_techniques = []
        
        # Apply selected augmentations
        for technique in selected_techniques:
            augmented, technique_info = technique(augmented, sr)
            applied_techniques.append(technique_info)
        
        return augmented, applied_techniques

    def create_spectrogram(self, audio, sr, output_path):
        """Generate and save regular spectrogram visualization"""
        plt.figure(figsize=(10, 4))
        
        # Create spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)),
            ref=np.max
        )
        
        # Display spectrogram
        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            cmap='viridis'
        )
        
        # Add colorbar
        plt.colorbar(format='%+2.0f dB')
        
        # Add title
        plt.title('Spectrogram')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and close
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def save_audio(self, audio, sr, save_path):
        """Save audio file"""
        sf.write(str(save_path), audio, sr) 
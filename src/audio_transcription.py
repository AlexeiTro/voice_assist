import whisper
import numpy as np
import soundfile as sf
from scipy import signal
import argparse
import json
import os
from typing import Dict, Union, List, Any, Optional


class AudioTranscriber:
    def __init__(self, model_size: str = "large", language: str = "de"):
        """
        Initialize the transcriber with specified model size and language.
        
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large")
            language: Language code ("de" for German, "en" for English, etc.)
        """
        self.model = whisper.load_model(model_size)
        self.language = language
    
    def preprocess_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Apply noise reduction and filtering to improve audio quality.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the processed audio (if None, creates a temp file)
            
        Returns:
            Path to the processed audio file
        """
        # Create output path if not provided
        if output_path is None:
            filename, ext = os.path.splitext(audio_path)
            output_path = f"{filename}_processed{ext}"
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Apply a high-pass filter to remove low-frequency noise
        sos = signal.butter(10, 100, 'hp', fs=sample_rate, output='sos')
        audio_filtered = signal.sosfilt(sos, audio_data)
        
        # Apply noise reduction using spectral gating
        # A simple implementation - could be replaced with more advanced methods
        # like librosa.effects.preemphasis or noisereduce library
        stft = np.abs(np.fft.rfft(audio_filtered))
        noise_threshold = np.percentile(stft, 20)  # Estimate noise level
        mask = stft > noise_threshold * 2  # Create a binary mask
        stft_denoised = stft * mask  # Apply mask
        audio_denoised = np.fft.irfft(stft_denoised * np.exp(1j * np.angle(np.fft.rfft(audio_filtered))))
        
        # Normalize audio volume
        audio_normalized = audio_denoised / (np.max(np.abs(audio_denoised)) + 1e-8)
        
        # Save processed audio
        sf.write(output_path, audio_normalized, sample_rate)
        
        return output_path
        
    def transcribe(self, 
                  audio_path: str,
                  apply_preprocessing: bool = True,
                  return_segments: bool = False) -> Dict[str, Any]:
        """
        Transcribe audio file with optional preprocessing.
        
        Args:
            audio_path: Path to the audio file
            apply_preprocessing: Whether to apply audio preprocessing
            return_segments: Whether to return time-stamped segments
            
        Returns:
            Dictionary with transcription results
        """
        if apply_preprocessing:
            processed_path = self.preprocess_audio(audio_path)
            audio_path = processed_path
            
        # Transcribe with Whisper, specifying German language
        options = {
            "language": self.language,
            "task": "transcribe",
            "fp16": False  # Set to True for GPU acceleration if available
        }
        
        result = self.model.transcribe(audio_path, **options)
        
        if not return_segments:
            return {"text": result["text"]}
        else:
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"]
            }
    
    def save_transcript(self, result: Dict[str, Any], output_path: str) -> None:
        """Save transcription results to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            if "segments" in result:
                # Format with timestamps if segments are available
                for segment in result["segments"]:
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"]
                    f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
            else:
                # Simple text output
                f.write(result["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files with enhanced features")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--language", type=str, default="de", help="Language code (de for German)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--segments", action="store_true", help="Include time segments")
    parser.add_argument("--no-preprocessing", action="store_true", help="Skip audio preprocessing")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    transcriber = AudioTranscriber(model_size=args.model, language=args.language)
    
    print(f"Transcribing {args.audio}...")
    result = transcriber.transcribe(
        args.audio, 
        apply_preprocessing=not args.no_preprocessing,
        return_segments=args.segments or args.json
    )
    
    # Handle output
    if args.output:
        if args.json:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            transcriber.save_transcript(result, args.output)
        print(f"Transcript saved to {args.output}")
    else:
        # Just print the text if no output file specified
        print("\nTranscription:")
        print(result["text"])
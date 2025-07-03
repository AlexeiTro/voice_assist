import os
import tempfile
import logging
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal
import whisper

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("AudioTranscriber")


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "medium",
        language: str = "de",
        device: Optional[str] = None,
    ):
        """
        Initialize the transcriber with specified model size, language, and device.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large").
            language: Language code ("de", "en", etc.).
            device: Torch device string (e.g., "cuda", "cpu"). If None, auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading Whisper model '{model_size}' on {device}")
        self.model = whisper.load_model(model_size, device=device)
        self.language = language

    def _apply_highpass(self, audio: np.ndarray, sr: int, cutoff: int = 100) -> np.ndarray:
        sos = signal.butter(10, cutoff, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, audio)
        
    def _apply_bandpass(self, audio: np.ndarray, sr: int, low_cutoff: int = 300, high_cutoff: int = 3400) -> np.ndarray:
        """Apply bandpass filter to focus on speech frequencies"""
        sos = signal.butter(10, [low_cutoff, high_cutoff], 'bp', fs=sr, output='sos')
        return signal.sosfilt(sos, audio)
        
    def _apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance higher frequencies"""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
        
    def _compress_dynamic_range(self, audio: np.ndarray, threshold: float = -20, ratio: float = 4) -> np.ndarray:
        """Apply dynamic range compression to make quiet parts louder and loud parts quieter"""
        # Convert to dB
        db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Apply compression
        mask = db > threshold
        db[mask] = threshold + (db[mask] - threshold) / ratio
        
        # Convert back to amplitude
        return np.sign(audio) * 10 ** (db / 20)

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction with advanced parameters"""
        return nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,  # Non-stationary noise estimation
            prop_decrease=0.75,
            n_fft=1024,
            win_length=512,
            n_std_thresh_stationary=1.5
        )

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio)) + 1e-8
        return audio / peak
        
    def _spectral_subtraction(self, audio: np.ndarray, sr: int, frame_len: int = 512, hop_len: int = 128, alpha: float = 2) -> np.ndarray:
        """Apply spectral subtraction for further noise reduction"""
        from scipy import fft
        
        # Estimate noise from first 500ms
        noise_len = min(int(sr * 0.5), len(audio) // 4)  # Use at most 1/4 of audio for noise estimation
        noise_sample = audio[:noise_len]
        noise_spec = np.abs(fft.rfft(noise_sample))
        noise_power = np.mean(noise_spec ** 2)
        
        # Process frames
        result = np.zeros_like(audio)
        window = np.hanning(frame_len)
        
        for i in range(0, len(audio) - frame_len, hop_len):
            frame = audio[i:i+frame_len] * window
            spec = fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)
            
            # Subtract noise
            mag_sq = mag ** 2
            power_diff = mag_sq - alpha * noise_power
            power_diff = np.maximum(power_diff, 0.01 * mag_sq)
            
            # Reconstruct
            mag_new = np.sqrt(power_diff)
            spec_new = mag_new * np.exp(1j * phase)
            frame_new = np.real(fft.irfft(spec_new))
            
            # Overlap-add
            result[i:i+frame_len] += frame_new * window
        
        # Normalize for overlapping windows
        # Count how many windows overlap at each point
        window_count = np.zeros_like(audio)
        for i in range(0, len(audio) - frame_len, hop_len):
            window_count[i:i+frame_len] += window
        
        # Avoid division by zero
        window_count = np.maximum(window_count, 1e-8)
        
        # Normalize
        result = result / window_count
        
        return result

    def preprocess_audio(
        self,
        audio_path: str,
        debug: bool = False,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Preprocess audio: mono conversion, high-pass filter, denoising, normalization.

        Args:
            audio_path: Input file path.
            debug: If True, logs intermediate steps.
            output_path: If provided, saves to this path; otherwise, uses a temp file.

        Returns:
            Path to processed audio file.
        """
        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            logger.error(f"Failed to read '{audio_path}': {e}")
            raise

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            if debug:
                logger.debug("Converted stereo to mono.")

        # Apply pre-emphasis
        audio = self._apply_preemphasis(audio)
        if debug:
            logger.debug("Applied pre-emphasis filter.")

        # High-pass filter
        audio = self._apply_highpass(audio, sr)
        if debug:
            logger.debug("Applied high-pass filter.")
            
        # Bandpass filter focusing on speech frequencies
        audio = self._apply_bandpass(audio, sr)
        if debug:
            logger.debug("Applied bandpass filter (300-3400Hz).")

        # Denoise
        audio = self._denoise(audio, sr)
        if debug:
            logger.debug("Applied advanced noise reduction.")
            
        # Spectral subtraction for additional noise reduction
        audio = self._spectral_subtraction(audio, sr)
        if debug:
            logger.debug("Applied spectral subtraction.")
            
        # Dynamic range compression
        audio = self._compress_dynamic_range(audio)
        if debug:
            logger.debug("Applied dynamic range compression.")

        # Normalize
        audio = self._normalize(audio)
        if debug:
            logger.debug("Normalized audio volume.")

        # Write to temp or specified path
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()
        else:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, "processed_audio.wav")

        sf.write(output_path, audio, sr)
        if debug:
            logger.debug(f"Processed audio saved to {output_path}.")
        return output_path

    def transcribe_file(
        self,
        audio_path: str,
        preprocess: bool = True,
        debug: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to input audio file.
            preprocess: Whether to apply preprocessing.
            debug: If True, enables verbose logging in preprocessing.

        Returns:
            Dict with keys: 'text', optionally 'segments', 'language'.
        """
        if preprocess:
            if debug:
                audio_path = self.preprocess_audio(audio_path, debug=debug, output_path=output_path)
            else:
                audio_path = self.preprocess_audio(audio_path, debug=debug)

        options = dict(
            language=self.language,
            task="transcribe",
            fp16=(self.device == "cuda"),
        )
        logger.info(f"Starting transcription with options: {options}")
        result = self.model.transcribe(audio_path, **options)

        # always include raw text
        output = {"text": result.get("text", "")}
        logger.info("Transcription complete.")
        return output

    async def transcribe_bytes(
        self,
        audio_bytes: Union[bytes, bytearray],
        preprocess: bool = True,
        return_segments: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio from in-memory bytes (for FastAPI endpoints).

        Args:
            audio_bytes: Raw audio data in memory.
            preprocess: Whether to preprocess.
            return_segments: If True, includes segments.
            debug: Verbose preprocessing logs.

        Returns:
            Same format as transcribe_file.
        """
        # Write bytes to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        try:
            return self.transcribe_file(
                tmp.name,
                preprocess=preprocess,
                return_segments=return_segments,
                debug=debug,
            )
        finally:
            os.unlink(tmp.name)


# Example CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio with Whisper and preprocessing"
    )
    parser.add_argument(
        "--audio", type=str, default="/home/tlr4fe/git/voice_assist/data/test_audios/poor-audio.ogg", help="Path to input audio file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Language code"
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Skip audio preprocessing",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output file path (text or JSON if --segments)",
    )
    args = parser.parse_args()

    # Create timestamped directory for output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    transcript_dir = os.path.join(args.output, f"transcripts_{timestamp}")
    os.makedirs(transcript_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {transcript_dir}")

    if args.debug:
        logger.setLevel(logging.DEBUG)

    transcriber = AudioTranscriber(
        model_size=args.model,
        language=args.language,
    )
    logger.info(f"Transcribing file: {args.audio}")
    res = transcriber.transcribe_file(
        args.audio,
        preprocess=not args.no_preprocessing,
        debug=args.debug,
        output_path=transcript_dir if args.debug else None,
    )

    # Save or print
    if args.output:
        # Save transcript
        output_file = os.path.join(transcript_dir, "audio_transcript.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(res["text"])
        logger.info(f"Transcript saved to {output_file}")
    else:
        # Still print to console
        print(res["text"])

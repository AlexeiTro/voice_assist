import os
import tempfile
import logging
import json
from typing import Dict, Any, Optional, Union

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

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return nr.reduce_noise(y=audio, sr=sr)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio)) + 1e-8
        return audio / peak

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

        # High-pass filter
        audio = self._apply_highpass(audio, sr)
        if debug:
            logger.debug("Applied high-pass filter.")

        # Denoise
        audio = self._denoise(audio, sr)
        if debug:
            logger.debug("Applied noise reduction.")

        # Normalize
        audio = self._normalize(audio)
        if debug:
            logger.debug("Normalized audio volume.")

        # Write to temp or specified path
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        sf.write(output_path, audio, sr)
        if debug:
            logger.debug(f"Processed audio saved to {output_path}.")
        return output_path

    def transcribe_file(
        self,
        audio_path: str,
        preprocess: bool = True,
        return_segments: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to input audio file.
            preprocess: Whether to apply preprocessing.
            return_segments: If True, includes time-stamped segments.
            debug: If True, enables verbose logging in preprocessing.

        Returns:
            Dict with keys: 'text', optionally 'segments', 'language'.
        """
        if preprocess:
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
        if return_segments:
            output["segments"] = result.get("segments", [])
            output["language"] = result.get("language", self.language)
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
        "--language", type=str, default="de", help="Language code"
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Skip audio preprocessing",
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Return time-stamped segments",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (text or JSON if --segments)",
    )
    args = parser.parse_args()

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
        return_segments=args.segments,
        debug=args.debug,
    )

    # Save or print
    if args.output:
        if args.segments:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON transcript saved to {args.output}")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(res["text"])
            logger.info(f"Transcript saved to {args.output}")
    else:
        print(res["text"])

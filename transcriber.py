#!/usr/bin/env python3
import argparse
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import sys
from faster_whisper import WhisperModel

def record_audio(duration: float, sample_rate: int = 44100) -> str:
    """Record audio and save to temporary WAV file."""
    # print(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()  # Wait until recording is finished
    
    # Save to temporary file
    temp_file = tempfile.mktemp(suffix='.wav')
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    return temp_file

def transcribe_audio(audio_file: str, model_type: str = "base", use_gpu: bool = False) -> str:
    """Transcribe audio file using Faster Whisper."""
    # Set device and compute type based on GPU availability
    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "int8"
    
    # Initialize model
    model = WhisperModel(model_type, device=device, compute_type=compute_type)
    
    # Transcribe
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language="en",
        condition_on_previous_text=False
    )
    
    # Combine all segments into one string
    return " ".join(segment.text for segment in segments)

def main():
    parser = argparse.ArgumentParser(description="Record audio and transcribe it to text")
    parser.add_argument(
        "duration",
        type=float,
        help="Duration to record in seconds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
        help="Whisper model size to use"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for transcription"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (if not specified, prints to stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        # Record audio
        audio_file = record_audio(args.duration)
        
        # Transcribe
        text = transcribe_audio(audio_file, args.model, args.gpu)
        
        # Output handling
        if args.output:
            with open(args.output, 'w') as f:
                f.write(text)
        else:
            print(text)
            
    finally:
        # Cleanup temporary file
        if 'audio_file' in locals():
            os.remove(audio_file)

if __name__ == "__main__":
    main()
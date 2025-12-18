import re
import threading
import queue
import logging
from pathlib import Path
from functools import partial
import sounddevice as sd
import numpy as np
from kokoro import KPipeline
from src.interfaces.speech_output import ISpeechOutput
from src.implementations.tts.speech_pipeline import SpeechPipeline

def _kokoro_worker(text_queue, audio_queue, pipeline, voice, speed):
    """
    Worker function that runs in the background thread.
    Uses the Kokoro pipeline to generate audio and put it into the audio queue.
    """
    logging.info("Kokoro worker started")
    
    while True:
        try:
            item = text_queue.get()
            if item is None:
                # Add None to audio queue to signal end (though we keep stream open)
                text_queue.task_done()
                break
            
            logging.info(f"Synthesizing: {item}")
            print(f"TTS: {item}", flush=True)
            
            # Generate audio
            # pipeline returns a generator of (graphemes, phonemes, audio)
            generator = pipeline(item, voice=voice, speed=speed, split_pattern=r'\n+')
            
            for i, (gs, ps, audio) in enumerate(generator):
                # audio is a numpy array (float32)
                # Sample rate is 24000 for Kokoro
                if audio is not None and len(audio) > 0:
                    audio_queue.put(audio)
                
            text_queue.task_done()
            
        except Exception as e:
            logging.error(f"Kokoro worker error: {e}")
            text_queue.task_done()

class KokoroTTSOutput(ISpeechOutput):
    def __init__(self):
        # Initialize pipeline (this might download weights)
        print("Initializing Kokoro pipeline... (this may download model weights)")
        # lang_code='a' is for American English
        self.pipeline = KPipeline(lang_code='a') 
        self.voice = 'af_heart'
        self.speed = 1.25
        
        # Audio playback state
        self.audio_queue = queue.Queue()
        self._current_chunk = None
        self._chunk_offset = 0
        
        # Initialize sounddevice stream
        self.stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()
        
        # Get the singleton text pipeline
        self._speech_pipeline = SpeechPipeline()
        
        # Configure backend
        # Pass both queues to the worker
        worker_func = partial(
            _kokoro_worker, 
            audio_queue=self.audio_queue, 
            pipeline=self.pipeline, 
            voice=self.voice,
            speed=self.speed
        )
        self._speech_pipeline.set_backend(worker_func)
        self._speech_pipeline.start()
        
        self._buffer = ""
        self._buffer_lock = threading.Lock()
        self._flush_timer = None

    def _audio_callback(self, outdata, frames, time, status):
        """
        Callback for sounddevice OutputStream.
        Fills the output buffer from the audio_queue.
        """
        if status:
            logging.warning(f"Audio stream status: {status}")
        
        filled = 0
        while filled < frames:
            if self._current_chunk is None:
                try:
                    # Get next chunk from queue
                    # Use timeout=0 (get_nowait) because we are in an audio callback
                    self._current_chunk = self.audio_queue.get_nowait()
                    self._chunk_offset = 0
                except queue.Empty:
                    # No more audio ready, fill remainder with silence
                    outdata[filled:].fill(0)
                    break
            
            # Copy data from current chunk
            chunk_remaining = len(self._current_chunk) - self._chunk_offset
            needed = frames - filled
            to_copy = min(chunk_remaining, needed)
            
            # Kokoro returns sequences of floats, sounddevice expects (frames, channels)
            # We use 1 channel so we slice it into outdata[:, 0]
            outdata[filled:filled+to_copy, 0] = self._current_chunk[self._chunk_offset:self._chunk_offset+to_copy]
            
            filled += to_copy
            self._chunk_offset += to_copy
            
            if self._chunk_offset >= len(self._current_chunk):
                self._current_chunk = None
        
    def speak(self, text: str) -> None:
        """Enqueue text for speech."""
        with self._buffer_lock:
            # Cancel any pending flush
            if self._flush_timer:
                self._flush_timer.cancel()
            
            self._buffer += text
            self._process_buffer()
            
            # Schedule new flush if buffer still has content
            if self._buffer.strip():
                self._flush_timer = threading.Timer(0.5, self._flush_internal)
                self._flush_timer.start()

    def _flush_internal(self) -> None:
        """Force output of any buffered text."""
        with self._buffer_lock:
            if self._buffer.strip():
                self._speech_pipeline.enqueue(self._buffer)
            self._buffer = ""
        
    def _process_buffer(self) -> None:
        """
        Segments buffer into sentences and enqueues complete ones.
        """
        parts = re.split(r'(?<=[.!?])\s+', self._buffer)
        
        if len(parts) > 1:
            for part in parts[:-1]:
                if part.strip():
                    self._speech_pipeline.enqueue(part)
            
            self._buffer = parts[-1]

    def stop(self):
        """Clean up resources."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self._speech_pipeline.stop()

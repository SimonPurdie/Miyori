import json
import logging
from pathlib import Path
from functools import partial
from src.interfaces.speech_output import ISpeechOutput
from src.implementations.tts.speech_pipeline import SpeechPipeline

def _pyttsx_worker(q, rate):
    """
    Worker function that runs in the background thread.
    Owns the pyttsx3 engine instance and keeps a persistent event loop.
    """
    import pyttsx3
    import time
    import queue
    import uuid
    
    try:
        # Initialize engine only once in this thread
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        
        # Setup callback for task completion
        def on_finished(name, completed):
            try:
                q.task_done()
            except ValueError:
                pass
                
        engine.connect('finished-utterance', on_finished)
        
        # Start the loop without blocking
        engine.startLoop(False)
        logging.info(f"pyttsx3 initialized with rate={rate} in persistent loop mode")
        
        while True:
            # Pump the engine's event loop
            engine.iterate()
            
            # Check for new items without blocking the loop
            try:
                item = q.get_nowait()
                if item is None:
                    # Poison pill
                    q.task_done()
                    break
                
                logging.info(f"Speaking: {item}")
                print(f"Speaking: {item}")
                engine.say(item, name=str(uuid.uuid4()))
                
            except queue.Empty:
                # Small sleep to prevent 100% CPU usage while idle
                time.sleep(0.01)
                
        # Clean up
        engine.endLoop()
        
    except Exception as e:
        logging.error(f"pyttsx3 worker crashed: {e}")
        # Ensure we don't leave the queue hanging if we crash
        try:
             while True:
                 q.get_nowait()
                 q.task_done()
        except:
             pass

class PyttsxOutput(ISpeechOutput):
    """
    Pyttsx3 implementation of ISpeechOutput.
    Delegates actual work to the centralized SpeechPipeline.
    """
    def __init__(self):
        # Determine config path (relative to this file)
        # src/implementations/tts/pyttsx_output.py -> ... -> config.json
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        
        rate = 180
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                tts_config = config.get("speech_output", {})
                rate = tts_config.get("rate", 180)
            except Exception as e:
                logging.warning(f"Could not load TTS config: {e}. Using default.")
        
        # Get the singleton pipeline
        self.pipeline = SpeechPipeline()
        
        # Configure and start if not already running
        # We pass the partial function with the config loaded 
        # (this only takes effect if set_backend is successful, i.e. first caller)
        worker_func = partial(_pyttsx_worker, rate=rate)
        self.pipeline.set_backend(worker_func)
        self.pipeline.start()

    def speak(self, text: str) -> None:
        """Enqueue text for speech."""
        self.pipeline.enqueue(text)

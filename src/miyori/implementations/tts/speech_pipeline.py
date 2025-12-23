import threading
import queue
import logging
from typing import Callable, Optional

class SpeechPipeline:
    """
    A singleton manager for the speech output pipeline.
    It manages a background worker thread and a queue of text to speak,
    ensuring a single serialized event loop for the TTS engine.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SpeechPipeline, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._backend_func: Optional[Callable[[queue.Queue], None]] = None
        self._initialized = True

    def set_backend(self, backend_func: Callable[[queue.Queue], None]):
        """
        Sets the backend function that will run in the worker thread.
        This function should take the queue as an argument and process items from it.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                 logging.warning("Attempting to set backend while pipeline is running. Ignoring.")
                 return
            self._backend_func = backend_func

    def start(self):
        """Starts the worker thread if it's not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            
            if not self._backend_func:
                raise RuntimeError("No backend configured for SpeechPipeline. Call set_backend first.")

            self._thread = threading.Thread(target=self._worker, daemon=True, name="SpeechPipelineWorker")
            self._thread.start()
            logging.info("SpeechPipeline worker started.")

    def _worker(self):
        """Internal wrapper to run the backend function."""
        try:
            if self._backend_func:
                self._backend_func(self._queue)
        except Exception as e:
            logging.error(f"SpeechPipeline worker crashed: {e}", exc_info=True)

    def enqueue(self, text: str):
        """
        Enqueues text to be spoken.
        
        Args:
            text: The text to speak.
        """
        if not text:
            return
        self._queue.put(text)

    def stop(self):
        """Stops the worker thread gracefully by sending a poison pill."""
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_for_completion(self):
        """Blocks until all items in the queue have been processed."""
        self._queue.join()

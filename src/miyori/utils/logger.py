import sys
import os
import datetime
import io
from pathlib import Path
from contextlib import contextmanager

class Tee:
    """
    A helper class that replicates data written to a stream 
    into another stream (like the Unix 'tee' command).
    """
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass # Avoid crashing if a stream is closed

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def add_stream(self, stream):
        if stream not in self.streams:
            self.streams.append(stream)

    def remove_stream(self, stream):
        if stream in self.streams:
            self.streams.remove(stream)

# Global trackers for the active Tee instances
_stdout_tee = None
_stderr_tee = None

def setup_logging():
    """
    Initializes a log file in the \logs directory and redirects 
    sys.stdout and sys.stderr to it, while also maintaining output 
    to the terminal.
    """
    global _stdout_tee, _stderr_tee

    # Define logs directory at project root
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs" / "terminal"
    
    # Create logs directory if it doesn't exist
    if not logs_dir.exists():
        os.makedirs(logs_dir)
        
    # Generate filename: Miyori_YYYYMMDD_HHMMSS_<PID>.log
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_filename = f"Miyori_{timestamp}_{pid}.log"
    log_path = logs_dir / log_filename
    
    # Open log file in append mode (though it's a new file)
    log_file = open(log_path, "a", encoding="utf-8")
    
    # Redirect stdout and stderr using the Tee class
    _stdout_tee = Tee(sys.stdout, log_file)
    _stderr_tee = Tee(sys.stderr, log_file)
    
    sys.stdout = _stdout_tee
    sys.stderr = _stderr_tee
    
    print(f"--- Process started: {now.isoformat()} ---")
    print(f"--- Logging to: {log_path} ---")

@contextmanager
def capture_session():
    """
    Context manager to capture all stdout/stderr output within a block.
    Integrates with the existing Tee loggers if they are active.
    """
    buffer = io.StringIO()
    
    if _stdout_tee:
        _stdout_tee.add_stream(buffer)
    if _stderr_tee:
        _stderr_tee.add_stream(buffer)
        
    try:
        yield buffer
    finally:
        if _stdout_tee:
            _stdout_tee.remove_stream(buffer)
        if _stderr_tee:
            _stderr_tee.remove_stream(buffer)

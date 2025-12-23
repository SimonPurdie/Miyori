import os
import datetime
import json
from pathlib import Path
from src.utils.config import Config

class MemoryLogger:
    """Utility for logging memory decisions and metrics for observability."""
    
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent
        self.log_dir = project_root / "logs"
        if not self.log_dir.exists():
            os.makedirs(self.log_dir)
        self.log_file = self.log_dir / "memory.log"

        # Cache verbose logging setting to avoid repeated config lookups
        self._verbose_checked = False
        self._is_verbose = False

    def log_event(self, event_type: str, details: dict = None , level: str = "DEBUG"):

        if details is None:
            details = {}
            
        """Log a memory-related event with its details."""
        if level == "DEBUG":
            if not self._verbose_checked:
                self._is_verbose = Config.data.get("memory", {}).get("verbose_logging", False)
                self._verbose_checked = True
            if not self._is_verbose:
                return

        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event": event_type.upper(),
            "details": details
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            import sys
            sys.stderr.write(f"Failed to write to memory log: {e}\n")

# Global instance for easy access
memory_logger = MemoryLogger()

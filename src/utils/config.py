import json
from pathlib import Path
from typing import Any

class Config:
    data = {}
    _root = None

    @classmethod
    def load(cls):
        """Finds config.json relative to this file and loads it into memory."""
        if not cls.data:
            # Move up from src/utils/ to project root
            cls._root = Path(__file__).resolve().parent.parent.parent
            config_path = cls._root / "config.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls.data = json.load(f)
            else:
                # Fallback or warning if config is missing
                print(f"Warning: config.json not found at {config_path}")
                cls.data = {}

    @classmethod
    def get(cls, key_path: str, default: Any = None) -> Any:
        """Retrieves values using dot notation (e.g., 'llm.model')"""
        keys = key_path.split('.')
        value = cls.data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

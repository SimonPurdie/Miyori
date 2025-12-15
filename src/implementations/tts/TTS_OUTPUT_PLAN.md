# Text-to-Speech Implementation Plan

## Step 1: Set up the file
- Create `pyttsx_output.py` in this directory
- Import required modules: `pyttsx3`, `json`, `ISpeechOutput` from interfaces

## Step 2: Define the class
```python
class PyttsTOutput(ISpeechOutput):
```

## Step 3: Implement __init__
```python
def __init__(self):
    # Open and load config.json from root (../../config.json)
    # Extract speech_output config section
    # Create self.engine = pyttsx3.init()
    # Set rate: self.engine.setProperty('rate', rate_from_config)
```

## Step 4: Implement speak()
```python
def speak(self, text: str) -> None:
    # Print f"Speaking: {text}"
    # self.engine.say(text)
    # self.engine.runAndWait()
```

## Interface Contract
```python
class ISpeechOutput(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech"""
```

## Config Keys (from config.json)
```json
{
  "speech_output": {
    "rate": 180
  }
}
```

## Notes
- `rate` = words per minute (default 180, range typically 100-300)
- Engine is reusable, initialize once in constructor
- Path to config.json from this file: `../../config.json`
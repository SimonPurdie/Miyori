# Speech Input Implementation Plan

## Step 1: Set up the file
- Create `google_speech_input.py` in this directory
- Import required modules: `speech_recognition as sr`, `json`, `pathlib.Path`, `ISpeechInput` from interfaces

## Step 2: Define the class
```python
class GoogleSpeechInput(ISpeechInput):
```

## Step 3: Implement __init__
```python
def __init__(self):
    # Get config path: Path(__file__).parent.parent.parent / "config.json"
    # Open and load config.json
    # Extract speech_input config section
    # Create sr.Recognizer()
    # Set self.recognizer.pause_threshold from config
    # Set self.recognizer.energy_threshold from config
```

## Step 4: Implement listen()
```python
def listen(self) -> str | None:
    # Open microphone: with sr.Microphone() as source:
    # Print "Calibrating microphone..."
    # Call self.recognizer.adjust_for_ambient_noise(source, duration=1)
    # Print "Listening..."
    # audio = self.recognizer.listen(source)
    # Print "Processing..."
    # Try to recognize: text = self.recognizer.recognize_google(audio)
    # Print f"You said: {text}"
    # Return text
```

## Interface Contract
```python
class ISpeechInput(ABC):
    @abstractmethod
    def listen(self) -> str | None:
        """Listen and return transcribed text or None"""
```

## Config Keys (from config.json)
```json
{
  "speech_input": {
    "pause_threshold": 2.0,
    "energy_threshold": 300
  }
}
```

## Notes
- `pause_threshold` = seconds of silence before considering speech complete
- `energy_threshold` = microphone sensitivity (lower = more sensitive)
- Use `Path(__file__).parent.parent.parent / "config.json"` for robust path resolution
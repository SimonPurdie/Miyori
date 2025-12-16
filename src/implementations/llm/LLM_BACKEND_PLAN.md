# LLM Backend Implementation Plan

## Step 1: Set up the file
- Create `google_ai_backend.py` in this directory
- Import required modules: `google.genai`, `json`, `pathlib.Path`, `ILLMBackend` from interfaces

## Step 2: Define the class
```python
class GoogleAIBackend(ILLMBackend):
```

## Step 3: Implement __init__
```python
def __init__(self):
    # Get config path: Path(__file__).parent.parent.parent / "config.json"
    # Open and load config.json
    # Extract llm config section
    # Get api_key and model from config
    # self.client = genai.Client(api_key=api_key)
    # self.model_name = model
```

## Step 4: Implement generate_stream()
```python
def generate_stream(self, prompt: str, on_chunk: Callable[[str], None]) -> None:
    # Print "Thinking..."
    # response = self.client.models.generate_content_stream(model=self.model_name, contents=prompt)
    # for chunk in response:
    #     if chunk.text:
    #         on_chunk(chunk.text)
```

## Interface Contract
```python
from typing import Callable

class ILLMBackend(ABC):
    @abstractmethod
    def generate_stream(self, prompt: str, on_chunk: Callable[[str], None]) -> None:
        """Generate AI response with streaming (calls on_chunk for each text chunk)"""
```

## Config Keys (from config.json)
```json
{
  "llm": {
    "api_key": "your_google_ai_key",
    "model": "gemini-2.0-flash-exp"
  }
}
```

## Notes
- API key obtainable from: https://aistudio.google.com/app/apikey
- `gemini-2.0-flash-exp` is fast and free-tier friendly
- Use `Path(__file__).parent.parent.parent / "config.json"` for robust path resolution
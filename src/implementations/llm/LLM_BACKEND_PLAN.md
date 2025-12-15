# LLM Backend Implementation Plan

## Step 1: Set up the file
- Create `google_ai_backend.py` in this directory
- Import required modules: `google.generativeai as genai`, `json`, `ILLMBackend` from interfaces

## Step 2: Define the class
```python
class GoogleAIBackend(ILLMBackend):
```

## Step 3: Implement __init__
```python
def __init__(self):
    # Open and load config.json from root (../../config.json)
    # Extract llm config section
    # Get api_key and model from config
    # genai.configure(api_key=api_key)
    # self.model = genai.GenerativeModel(model)
```

## Step 4: Implement generate()
```python
def generate(self, prompt: str) -> str:
    # Print "Thinking..."
    # response = self.model.generate_content(prompt)
    # Return response.text
```

## Interface Contract
```python
class ILLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate AI response"""
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
- Path to config.json from this file: `../../config.json`
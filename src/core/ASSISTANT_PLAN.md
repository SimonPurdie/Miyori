# Voice Assistant Core Implementation Plan

## Step 1: Set up the file
- Create `assistant.py` in this directory
- Import required modules: all three interfaces from `src/interfaces/`

## Step 2: Define the class
```python
class VoiceAssistant:
```

## Step 3: Implement __init__
```python
def __init__(self, 
             speech_input: ISpeechInput,
             speech_output: ISpeechOutput,
             llm: ILLMBackend):
    # Store all three dependencies as instance variables
    # self.speech_input = speech_input
    # self.speech_output = speech_output
    # self.llm = llm
```

## Step 4: Implement run()
```python
def run(self) -> None:
    # Print "Miyori starting up..."
    # while True:
    #     text = self.speech_input.listen()
    #     if text is None:
    #         continue  # Try again
    #     
    #     # Check for exit commands (case-insensitive)
    #     if any word in text.lower() for word in ['exit', 'quit', 'stop', 'goodbye']:
    #         self.speech_output.speak("Goodbye!")
    #         break
    #     
    #     response = self.llm.generate(text)
    #     self.speech_output.speak(response)
    # 
    # Print "Miyori shutting down..."
```

## Notes
- This class has NO knowledge of specific implementations
- It only uses the interface methods: listen(), speak(), generate()
- Exit commands: "exit", "quit", "stop", "goodbye" (case-insensitive)
- Loop continues even if listen() returns None (just try again)
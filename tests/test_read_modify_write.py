import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from miyori.tools.file_ops import file_operations

def test_read_modify_write_pattern():
    """Test the recommended file editing workflow."""
    test_file = "test_rmw.txt"
    initial_content = "Line 1\nLine 2\nLine 3"
    
    # 1. Write initial file
    print("Writing initial file...")
    res = file_operations(operation="write", path=test_file, content=initial_content)
    print(res)
    assert "Successfully wrote" in res
    
    # 2. Read it back
    print("\nReading file back...")
    res = file_operations(operation="read", path=test_file)
    print(res)
    assert initial_content in res
    
    # 3. Modify content in memory
    print("\nModifying content...")
    # Simulate LLM logic: add a line
    modified_content = initial_content + "\nLine 4"
    
    # 4. Write modified version
    print("Writing modified version...")
    res = file_operations(operation="write", path=test_file, content=modified_content)
    print(res)
    assert "Successfully wrote" in res
    
    # 5. Verify final state
    print("\nVerifying final state...")
    res = file_operations(operation="read", path=test_file)
    print(res)
    assert "Line 4" in res
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    elif os.path.exists(Path(test_file).resolve()):
         os.remove(Path(test_file).resolve())

    print("\n✓ Read-Modify-Write pattern verified successfully!")

if __name__ == "__main__":
    try:
        test_read_modify_write_pattern()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

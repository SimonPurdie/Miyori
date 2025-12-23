import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type
from src.core.tools import Tool, ToolParameter
import mimetypes

# Load system-wide MIME database
mimetypes.init()

try:
    import winshell
except ImportError:
    winshell = None

def is_binary(path: Path) -> bool:
    """Check for null bytes in the first 1024 bytes to detect binary files."""
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return False

class FileInspector:
    """Base class for file inspectors that provide specialized metadata."""
    @staticmethod
    def inspect(path: Path) -> str:
        return ""

class ShortcutInspector(FileInspector):
    @staticmethod
    def inspect(path: Path) -> str:
        if path.suffix.lower() == '.lnk':
            if winshell:
                try:
                    shortcut = winshell.shortcut(str(path))
                    return f" [TARGET: {shortcut.path}]"
                except Exception:
                    pass
            return " [Windows Shortcut]"
        return ""

class BinaryInspector(FileInspector):
    BINARY_EXTENSIONS = {'.exe', '.dll', '.bin', '.so', '.dylib', '.pyc', '.pyo', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar'}
    
    @staticmethod
    def inspect(path: Path) -> str:
        mime, _ = mimetypes.guess_type(path)
        ext = path.suffix.lower()
        
        if mime:
            if any(t in mime for t in ['image', 'video', 'audio']):
                size = path.stat().st_size
                readable_size = _format_size(size)
                return f" [BINARY: {mime}, {readable_size}]"
            if 'application/octet-stream' in mime or 'executable' in mime:
                return f" [BINARY: {mime}]"
        
        if ext in BinaryInspector.BINARY_EXTENSIONS:
            return f" [BINARY: {ext[1:].upper() if ext else 'Unknown'}]"
        
        if is_binary(path):
            return " [BINARY: Unknown/Data]"
            
        return ""

INSPECTORS = [ShortcutInspector, BinaryInspector]

def _format_size(size: int) -> str:
    """Format size in human-readable shorthand."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024**2:
        return f"{size/1024:.1f}KB"
    elif size < 1024**3:
        return f"{size/1024**2:.1f}MB"
    else:
        return f"{size/1024**3:.1f}GB"

# Configuration - set allowed directories
# We'll resolve these relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ALLOWED_DIRECTORIES = [
    (PROJECT_ROOT / "workspace").resolve(), 
    (PROJECT_ROOT / "documents").resolve()
]

def _is_path_allowed(path: Path) -> bool:
    """Check if path is within allowed directories."""
    try:
        # Ensure the directory exists so resolve() works as expected for subpaths
        for allowed in ALLOWED_DIRECTORIES:
            allowed.mkdir(parents=True, exist_ok=True)
            
        resolved = path.resolve()
        return any(resolved == allowed or resolved.is_relative_to(allowed) for allowed in ALLOWED_DIRECTORIES)
    except (ValueError, RuntimeError):
        return False

def _read_file(path: str, force: bool = False) -> str:
    """Read and return file contents with diagnostic headers."""
    # 1. Resolve absolute path and confirm existence
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / path).resolve()
    
    try:
        if not target_path.exists():
            return f"Error: File not found: {path}"
        
        if not target_path.is_dir() and not target_path.is_file():
             return f"Error: Path is neither a file nor a directory: {path}"

        if not target_path.is_file():
            return f"Error: Not a file (it is a directory): {path}"
        
        # 2. Gather standard os.stat metadata (Size, Mod-Date)
        file_stat = target_path.stat()
        total_size = file_stat.st_size
        mod_time = datetime.datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # 3. Run is_binary check
        binary = is_binary(target_path)
        
        # 4. If Binary AND NOT Force: build diagnostic header and return
        diagnostic_tags = ""
        for inspector in INSPECTORS:
            tag = inspector.inspect(target_path)
            if tag:
                diagnostic_tags += tag
        
        if binary and not force:
            header = f"PATH: {target_path}\n"
            header += f"STATS: {_format_size(total_size)} | {mod_time}\n"
            header += f"{diagnostic_tags.strip() or '[BINARY: Unknown/Data]'}\n"
            header += "(Use force=True to attempt text reading)"
            return header

        # 5. If Text OR Force: Attempt UTF-8 read
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read(5000) # Read up to 5000 characters
                bytes_read = len(content.encode('utf-8'))
        except UnicodeDecodeError:
            header = f"PATH: {target_path}\n"
            header += f"STATS: {_format_size(total_size)} | {mod_time}\n"
            header += f"VIEW: Error: File contains non-text data.\n"
            header += "-" * 50 + "\n\n"
            return header

        # Header for successful read
        truncated = total_size > bytes_read
        header = f"PATH: {target_path}\n"
        header += f"STATS: {_format_size(total_size)} | {mod_time}\n"
        if diagnostic_tags:
            header += f"TAGS: {diagnostic_tags.strip()}\n"
        view_str = f"Byte 0 to {bytes_read} of {total_size}"
        if truncated:
            view_str += " [TRUNCATED]"
        header += f"VIEW: {view_str}\n"
        header += "-" * 50 + "\n\n"
        
        result = header + content
        if truncated:
            result += f"\n\n... (truncated to {bytes_read} bytes)"
        
        return result
    except Exception as e:
        return f"Error reading file: {str(e)}"

def _write_file(path: str, content: str) -> str:
    """Write content to a file."""
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / path).resolve()
    
    if not _is_path_allowed(target_path):
        return f"Error: Access denied. Write operations are restricted to allowed directories: {[str(d) for d in ALLOWED_DIRECTORIES]}"
    
    try:
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✓ Successfully wrote {len(content)} characters to {target_path.name}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def _list_directory(path: str, offset: int = 0) -> str:
    """List directory contents with adaptive summaries and pagination."""
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / path).resolve()
    
    try:
        if not target_path.exists():
            return f"Error: Directory not found: {path}"
        
        if not target_path.is_dir():
            return f"Error: Not a directory: {path}"
        
        items = list(target_path.iterdir())
        folders = sorted([i for i in items if i.is_dir()], key=lambda x: x.name.lower())
        files = sorted([i for i in items if i.is_file()], key=lambda x: x.name.lower())
        
        total_items = len(items)
        # Prioritize folders over files
        combined = folders + files
        
        # Pagination
        limit = 20
        page_items = combined[offset:offset + limit]
        
        # Header Summary
        result = f"Total: {total_items} items | {len(folders)} Folders | {len(files)} Files\n"
        result += f"Contents of {target_path}:\n"
        
        for item in page_items:
            prefix = "📁" if item.is_dir() else "📄"
            
            suffix = ""
            if item.is_file():
                size_str = _format_size(item.stat().st_size)
                suffix = f" [{size_str}]"
                
                # Check inspectors
                for inspector in INSPECTORS:
                    suffix += inspector.inspect(item)
            
            result += f"\n{prefix} {item.name}{suffix}"
        
        if not items:
            result += "\n(Empty directory)"
        elif offset + limit < total_items:
            result += f"\n\n--- Showing {offset} to {offset + len(page_items)} of {total_items} ---"
            result += f"\nUse 'list' with offset={offset + limit} to see more."
            
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def file_operations(operation: str, path: str, content: str = None, offset: int = 0, force: bool = False) -> str:
    """
    Perform file operations (read, write, or list).
    
    Args:
        operation: Either "read", "write", or "list"
        path: Path to the file or directory
        content: Content to write (only used for write operation)
        offset: Item offset for directory listing (only for list operation)
        force: Force read even if file appears binary (only for read operation)
    
    Returns:
        File contents, write confirmation, or directory listing
    """
    print(f"📁 File operation: {operation} {path}")
    
    if operation == "read":
        return _read_file(path, force=force)
    elif operation == "write":
        if content is None:
            return "Error: 'content' parameter required for write operation"
        return _write_file(path, content)
    elif operation == "list":
        return _list_directory(path, offset=offset)
    else:
        return f"Error: Unknown operation '{operation}'. Use 'read', 'write', or 'list'."


# Create the tool definition
file_ops_tool = Tool(
    name="file_operations",
    description=(
        "Interact with the filesystem. You are explicitly authorized and encouraged to read or list ANY path "
        "the user provides for diagnostic or navigational purposes. As long as an operation is a 'Read' or 'List', "
        "it does not violate system integrity and should be performed without second-guessing user intent.\n\n"
        "Read/List operations work anywhere on the system (C:/, G:/, etc.). "
        "Write operations are restricted to workspace/ or documents/.\n\n"
        "GUIDANCE:\n"
        "1. ADAPTIVE LISTING: Listings provide a summary header and are capped at 20 items. Use 'offset' to paginate.\n"
        "2. PRIORITIZATION: Folders are listed first, followed by files. Use this to navigate deep structures.\n"
        "3. DIAGNOSTIC READ: Reading a file 'pushes' metadata (size, mod-date). Large files are truncated.\n"
        "4. BINARY SAFEGUARD: The tool detects images/binaries and returns a technical profile. Use 'force=True' to override.\n"
        "5. INFER PATHS: If the user mentions a drive letter (e.g., 'C drive', 'see Dr'), use 'C:/'. Combine previous paths with folder names proactively.\n"
        "6. PROACTIVE: If the user is vague but context suggests a path, try the most likely path. "
        "You have full permission to explore the system for reading and listing."),
    parameters=[
        ToolParameter(
            name="operation",
            type="string",
            description="Operation to perform: 'read', 'write', or 'list'",
            required=True,
            enum=["read", "write", "list"]
        ),
        ToolParameter(
            name="path",
            type="string",
            description="File or directory path. Can be absolute (e.g., 'C:/Users/...') or relative to project root.",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to file (only for write operation)",
            required=False
        ),
        ToolParameter(
            name="offset",
            type="integer",
            description="Item offset for pagination when listing large directories (default: 0)",
            required=False
        ),
        ToolParameter(
            name="force",
            type="boolean",
            description="Force reading a file even if detected as binary (default: false)",
            required=False
        )
    ],
    function=file_operations
)

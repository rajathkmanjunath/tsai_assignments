import mimetypes
from pathlib import Path

def get_file_type(filename):
    """Determine file type from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type.split('/')[0]
    return None

def ensure_directories(base_dir, subdirs):
    """Create necessary directories if they don't exist"""
    base_path = Path(base_dir)
    paths = {}
    
    # Create base directory
    base_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    for subdir in subdirs:
        path = base_path / subdir
        path.mkdir(exist_ok=True, parents=True)
        paths[subdir] = path
    
    return paths 
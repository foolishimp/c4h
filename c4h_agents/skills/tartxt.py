#!/usr/bin/env python3
# Path: tartxt.py

import os
import sys
import glob as glob_module
import fnmatch
import argparse
from typing import List, Tuple
import mimetypes

def get_file_metadata(file_path: str) -> Tuple[str, int, str]:
    """Get file metadata including MIME type, size, and last modified date."""
    mime_type, _ = mimetypes.guess_type(file_path)
    file_size = os.path.getsize(file_path)
    last_modified = os.path.getmtime(file_path)
    return mime_type or "application/octet-stream", file_size, last_modified

def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file based on its MIME type and extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # List of common text-based file extensions
    text_file_extensions = ['.dart', '.js', '.java', '.py', '.cpp', '.c', '.h', '.html', '.css', '.txt', '.md', '.sh', '.yml', '.yaml']

    if mime_type and (mime_type.startswith('text/') or mime_type in [
        'application/x-sh',
        'application/x-shellscript'
    ]):
        return True
    
    # Check the file extension as a fallback
    ext = os.path.splitext(file_path)[1].lower()
    return ext in text_file_extensions

def expand_path(path: str) -> List[str]:
    """Expand a path that may contain wildcards into a list of actual paths."""
    # Check if the path contains wildcards
    if '*' in path or '?' in path or '[' in path:
        expanded_paths = glob_module.glob(path)
        if not expanded_paths:
            return []  # Return empty list to indicate no matches
        return expanded_paths
    else:
        # Path doesn't contain wildcards
        return [path] if os.path.exists(path) else []

def is_excluded(file_path: str, exclusions: List[str]) -> bool:
    """Check if a file path matches any exclusion pattern."""
    # Normalize path for cross-platform comparison
    normalized_path = file_path.replace('\\', '/')
    
    # Check each exclusion pattern
    for pattern in exclusions:
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
            
    return False

def process_files(files: List[str], exclusions: List[str], include_binary: bool) -> str:
    """Process files and directories, excluding specified patterns."""
    output = "== Manifest ==\n"
    content = "\n== Content ==\n"
    processed_files = set()  # Track files to avoid duplicates

    for item in files:
        # Expand path if it contains wildcards
        expanded_items = expand_path(item)
        
        if not expanded_items:
            output += f"Warning: {item} does not exist, skipping.\n"
            continue
            
        for expanded_item in expanded_items:
            if os.path.isdir(expanded_item):
                for root, dirs, filenames in os.walk(expanded_item):
                    # Modify dirs in-place to avoid walking excluded directories
                    dirs[:] = [d for d in dirs if not is_excluded(os.path.join(root, d), exclusions)]
                    
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        normalized_path = file_path.replace('\\', '/')
                        
                        # Skip excluded files and already processed files
                        if is_excluded(normalized_path, exclusions) or normalized_path in processed_files:
                            continue
                            
                        output += f"{normalized_path}\n"
                        content += process_file(file_path, include_binary)
                        processed_files.add(normalized_path)
                        
            elif os.path.isfile(expanded_item):
                normalized_path = expanded_item.replace('\\', '/')
                
                # Skip excluded files and already processed files
                if is_excluded(normalized_path, exclusions) or normalized_path in processed_files:
                    continue
                    
                output += f"{normalized_path}\n"
                content += process_file(expanded_item, include_binary)
                processed_files.add(normalized_path)
            else:
                output += f"Warning: {expanded_item} does not exist, skipping.\n"

    return output + content

def process_file(file_path: str, include_binary: bool) -> str:
    """Process a single file, returning its content or a skip message for binary files."""
    mime_type, file_size, last_modified = get_file_metadata(file_path)
    
    output = f"\n== Start of File ==\n"
    output += f"File: {file_path}\n"
    output += f"File Type: {mime_type}\n"
    output += f"Size: {file_size} bytes\n"
    output += f"Last Modified: {last_modified}\n"

    if include_binary or is_text_file(file_path):
        output += "Contents:\n"
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                output += f.read()
        except Exception as e:
            output += f"Error reading file: {str(e)}\n"
        output += "\n== End of File ==\n"
    else:
        output += "Reason: Binary File, Skipped\n"
        output += "== End of File ==\n"

    return output

def get_incremented_filename(base_filename: str) -> str:
    """Generate an incremented filename if the file already exists."""
    name, ext = os.path.splitext(base_filename)
    counter = 0
    while True:
        new_filename = f"{name}_{counter:03d}{ext}" if counter > 0 else f"{name}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

def main():
    parser = argparse.ArgumentParser(description="Process and analyze files and directories.")
    parser.add_argument('-x', '--exclude', help="Glob patterns for files to exclude (comma-separated)", default="")
    parser.add_argument('-f', '--file', help="Output file name")
    parser.add_argument('-o', '--output', action='store_true', help="Output to stdout")
    parser.add_argument('--include-binary', action='store_true', help="Include content of binary files")
    parser.add_argument('items', nargs='+', help="Files and directories to process")

    args = parser.parse_args()

    exclusions = [pat.strip() for pat in args.exclude.split(',') if pat.strip()]
    
    # Print feedback about what we're processing
    print(f"Processing {len(args.items)} paths with {len(exclusions)} exclusion patterns")
    if exclusions:
        print(f"Exclusions: {', '.join(exclusions)}")
    
    result = process_files(args.items, exclusions, args.include_binary)

    if args.output:
        print(result)
    elif args.file:
        output_file = get_incremented_filename(args.file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Output written to {output_file}")
    else:
        print("Error: Either -f or -o must be specified.")
        sys.exit(1)

if __name__ == "__main__":
    main()
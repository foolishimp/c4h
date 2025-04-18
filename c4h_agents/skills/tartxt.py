#!/usr/bin/env python3
# src/skills/tartxt.py

import os
import sys
import glob
import argparse
import subprocess
import shlex # For safer command splitting if needed
from typing import List, Tuple
import mimetypes
from pathlib import Path

# --- Git Helper Functions ---

def run_git_command(command: List[str], cwd: str = None, verbose: bool = False) -> Tuple[bool, str]:
    """Runs a git command and returns (success_status, output_string)."""
    try:
        result = subprocess.run(
            ['git'] + command,
            capture_output=True,
            text=True,
            check=True, # Raise exception on non-zero exit code
            cwd=cwd or os.getcwd() # Run in specified directory or current
        )
        return True, result.stdout.strip()
    except FileNotFoundError:
        # Always print critical errors like git not found
        print("Error: 'git' command not found. Is Git installed and in your PATH?", file=sys.stderr)
        return False, "Git command not found"
    except subprocess.CalledProcessError as e:
        # Print git command errors only if verbose
        if verbose:
            print(f"Error running git command {' '.join(['git'] + command)}:", file=sys.stderr)
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False, e.stderr.strip()
    except Exception as e:
        # Always print unexpected errors
        print(f"Unexpected error running git command: {e}", file=sys.stderr)
        return False, str(e)

def get_git_history_diffs(file_path: str, history_count: int, repo_root: str, verbose: bool = False) -> str:
    """Gets the last N diffs for a file."""
    if history_count <= 0:
        return ""

    history_output = "\n== History Diffs (Last {} Commits) ==\n".format(history_count)

    log_command = [
        'log',
        f'--max-count={history_count + 1}',
        '--pretty=format:%H',
        '--follow',
        '--',
        file_path
    ]
    # Pass verbose flag down
    success, hashes_str = run_git_command(log_command, cwd=repo_root, verbose=verbose)

    if not success or not hashes_str:
        history_output += f"--- No Git history found for {file_path} ---\n"
        history_output += "== End History Diffs ==\n"
        return history_output

    hashes = hashes_str.splitlines()

    if len(hashes) < 2:
        history_output += f"--- Not enough history (found {len(hashes)} commits) to generate diffs for {file_path} ---\n"
        history_output += "== End History Diffs ==\n"
        return history_output

    num_diffs_to_get = min(history_count, len(hashes) - 1)
    for i in range(num_diffs_to_get):
        commit_new = hashes[i]
        commit_old = hashes[i+1]

        history_output += f"\n--- Diff: {commit_old[:7]}..{commit_new[:7]} ---\n"

        diff_command = [
            'diff',
            f'{commit_old}..{commit_new}',
            '--',
            file_path
        ]
        # Pass verbose flag down
        success, diff_str = run_git_command(diff_command, cwd=repo_root, verbose=verbose)

        if success:
            history_output += diff_str + "\n"
        else:
            history_output += f"--- Error retrieving diff between {commit_old[:7]} and {commit_new[:7]} ---\n"
            # Display error message only if verbose, otherwise just note the failure
            if verbose:
                history_output += f"--- Error message: {diff_str} ---\n"


    history_output += "== End History Diffs ==\n"
    return history_output


# --- Original File Processing Functions (Modified) ---

def get_file_metadata(file_path: str) -> Tuple[str, int, float]: # Changed type hint for mtime
    """Get file metadata including MIME type, size, and last modified date."""
    mime_type, _ = mimetypes.guess_type(file_path)
    try:
        file_size = os.path.getsize(file_path)
        last_modified = os.path.getmtime(file_path)
    except OSError: # Handle case where file might disappear between listing and statting
        return "application/octet-stream", 0, 0.0
    return mime_type or "application/octet-stream", file_size, last_modified

def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file based on its MIME type and extension."""
    mime_type, _ = mimetypes.guess_type(file_path)

    text_file_extensions = [
        '.dart', '.js', '.java', '.py', '.cpp', '.c', '.h', '.html', '.css',
        '.txt', '.md', '.sh', '.yml', '.yaml', '.json', '.tsx', '.ts', '.sql',
        '.xml', '.toml', '.ini', '.cfg', '.conf', '.log', '.env', '.gitignore',
        '.dockerfile', '.tf', '.tfvars', '.hcl' # Added more common text types
    ]

    if mime_type:
        if mime_type.startswith('text/') or \
           mime_type in ['application/json', 'application/yaml', 'application/x-yaml',
                           'application/xml', 'application/javascript', 'application/typescript',
                           'application/x-sh', 'application/x-shellscript']:
            return True

    ext = os.path.splitext(file_path)[1].lower()
    return ext in text_file_extensions

def process_files(files: List[str], exclusions: List[str], include_binary: bool, history_count: int, repo_root: str, verbose: bool = False) -> str:
    """Process files and directories, excluding specified patterns and adding history."""
    output = "== Manifest ==\n"
    content = "\n== Content ==\n"
    processed_files = set()

    items_to_process = list(files)

    while items_to_process:
        item_path_str = items_to_process.pop(0)
        try:
            item_path = Path(item_path_str).resolve()
        except Exception as e:
             # Always print errors related to resolving input paths
             print(f"Warning: Could not resolve path {item_path_str}, skipping. Error: {e}", file=sys.stderr)
             continue

        # Check exclusions against the absolute path
        if any(item_path.match(pat) for pat in exclusions):
             # Print exclusion messages only if verbose
             if verbose: print(f"Excluding (pattern match): {item_path_str}", file=sys.stderr)
             continue

        if item_path.is_dir():
            if verbose: print(f"Processing directory: {item_path_str}", file=sys.stderr)
            try:
                for entry in item_path.iterdir():
                    entry_path_str_for_queue = str(entry.resolve())
                    if not any(entry.match(pat) for pat in exclusions):
                        items_to_process.append(entry_path_str_for_queue)
                    else:
                        if verbose: print(f"Excluding sub-item: {entry}", file=sys.stderr)
            except OSError as e:
                 # Always print directory access errors
                 print(f"Warning: Cannot access directory {item_path_str}: {e}", file=sys.stderr)

        elif item_path.is_file():
            abs_path_str = str(item_path)
            if abs_path_str not in processed_files:
                if verbose: print(f"Processing file: {item_path_str}", file=sys.stderr)
                output += f"{abs_path_str}\n"
                # Pass verbose flag down
                content += process_file(abs_path_str, include_binary, history_count, repo_root, verbose)
                processed_files.add(abs_path_str)
        else:
             if verbose: print(f"Warning: {item_path_str} is not a regular file or directory, skipping.", file=sys.stderr)


    return output + content

def process_file(file_path: str, include_binary: bool, history_count: int, repo_root: str, verbose: bool = False) -> str:
    """Process a single file, returning its metadata, history diffs, and content."""
    mime_type, file_size, last_modified = get_file_metadata(file_path)

    output = f"\n== Start of File ==\n"
    output += f"File: {file_path}\n"
    output += f"File Type: {mime_type}\n"
    output += f"Size: {file_size} bytes\n"
    output += f"Last Modified: {last_modified}\n"

    # --- Add History Diffs ---
    if history_count > 0:
        relative_path = os.path.relpath(file_path, repo_root)
        # Pass verbose flag down
        output += get_git_history_diffs(relative_path, history_count, repo_root, verbose)
    # --- End History Diffs ---

    # --- Add Content ---
    output += "Contents:\n"
    if include_binary or is_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                output += f.read()
        except Exception as e:
            output += f"[Error reading file content: {e}]\n"
            # Always print file read errors
            print(f"Error reading file content for {file_path}: {e}", file=sys.stderr)
    else:
        output += "[Reason: Binary File, Skipped]\n"
    output += "\n== End of File ==\n"

    return output

def get_incremented_filename(base_filename: str) -> str:
    """Generate an incremented filename if the file already exists."""
    # (Implementation unchanged)
    name, ext = os.path.splitext(base_filename)
    counter = 0
    while True:
        new_filename = f"{name}_{counter:03d}{ext}" if counter > 0 else f"{name}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate file contents with metadata, optionally adding Git history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Process specific files and directories, excluding node_modules, output to file (quiet by default)
  tartxt.py src/ main.py config/ -x "**/node_modules/**,*.log" -f output.txt

  # Get files changed between two commits (excluding json), show 2 history diffs, enable verbose logging
  tartxt.py --git "diff --name-only HEAD~5 HEAD" -x "*.json" -H 2 -v -o

  # Get files changed since specific commit (incl. unstaged), default (quiet) logging
  tartxt.py --git "diff --name-only abc1234" -H 0 -f changed_files.txt

  # Process all files under packages, max 3 history diffs, enable verbose logging
  tartxt.py -H 3 /path/to/packages -x "**/__pycache__/**,**/*.pyc,**/node_modules/**" -v -f output.txt
  """
    )
    parser.add_argument('-x', '--exclude', help="Comma-separated glob patterns for exclusions", default="")
    parser.add_argument('-f', '--file', help="Output file name (will be incremented if exists)")
    parser.add_argument('-o', '--output', action='store_true', help="Output to stdout instead of a file")
    parser.add_argument('--include-binary', action='store_true', help="Include content of binary files")
    parser.add_argument('-g', '--git', help="Use 'git <command>' to list files instead of positional args")
    parser.add_argument('-H', '--history', type=int, default=0, help="Include N levels of git diff history (uses git)")
    # --- FIX: Changed quiet flag to verbose flag ---
    parser.add_argument('-v', '--verbose', action='store_true', help="Print status messages and verbose errors to stderr")
    # --- END FIX ---
    parser.add_argument('items', nargs='*', help="Files/dirs to process (ignored if --git used)")

    args = parser.parse_args()

    if not args.output and not args.file:
        parser.error("Error: Either -f/--file or -o/--output must be specified.")

    if args.git and args.items:
        # Print warning only if verbose
        if args.verbose: print("Warning: Positional arguments 'items' are ignored when --git flag is used.", file=sys.stderr)

    repo_root = os.getcwd()
    # Pass verbose flag down
    success, git_root_output = run_git_command(['rev-parse', '--show-toplevel'], verbose=args.verbose)
    if success:
        repo_root = git_root_output
    else:
        # Always print critical errors
        print(f"Warning: Could not determine Git repository root. Git commands will run from {repo_root}.", file=sys.stderr)
        if args.history > 0 or args.git:
            print("Error: Cannot use --history or --git features without a valid Git repository.", file=sys.stderr)
            sys.exit(1)


    files_to_process = []
    if args.git:
        if args.verbose: print(f"Using git command to determine files: git {args.git}", file=sys.stderr)
        git_cmd_parts = args.git.split()
        # Pass verbose flag down
        success, git_files_output = run_git_command(git_cmd_parts, cwd=repo_root, verbose=args.verbose)
        if not success:
             # Always print critical errors
            print(f"Error executing git command: git {' '.join(git_cmd_parts)}", file=sys.stderr)
            sys.exit(1)
        files_to_process = [str(Path(repo_root) / f) for f in git_files_output.splitlines() if f]

    elif args.items:
         files_to_process = [str(Path(item).resolve()) for item in args.items]
    else:
        parser.error("Error: You must provide input 'items' (files/dirs) or use the --git flag.")


    exclusions = [pat.strip() for pat in args.exclude.split(',') if pat.strip()]
    common_git_excludes = ["**/.git/**"]
    for exclude in common_git_excludes:
        if exclude not in exclusions:
            exclusions.append(exclude)

    # Print summary only if verbose
    if args.verbose:
        print(f"Processing items based on: {'Git command' if args.git else 'Positional arguments'}", file=sys.stderr)
        print(f"Exclusion patterns: {exclusions}", file=sys.stderr)
        print(f"History count: {args.history}", file=sys.stderr)
        print(f"Include binary: {args.include_binary}", file=sys.stderr)
        print(f"Repository root: {repo_root}", file=sys.stderr)

    # Pass verbose flag down
    result = process_files(files_to_process, exclusions, args.include_binary, args.history, repo_root, args.verbose)

    if args.output:
        print(result)
    elif args.file:
        output_path = Path(args.file).resolve()
        output_dir = output_path.parent
        if not output_dir.exists():
             output_dir.mkdir(parents=True, exist_ok=True)

        output_file = get_incremented_filename(str(output_path))
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            # Print final output location only if verbose
            if args.verbose: print(f"Output written to {output_file}")
        except Exception as e:
            # Always print actual errors
            print(f"Error writing output file {output_file}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
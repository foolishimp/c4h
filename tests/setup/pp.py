#!/usr/bin/env python3

import json
import codecs
import argparse
import glob
import sys
import os

# --- Enhanced Text Indenter ---

def simple_indent_text(text, indent_spaces=4, base_indent_level=0):
    """
    Applies indentation to text, aligning with base_indent_level for nested content.
    Args:
        text (str): Text block to indent.
        indent_spaces (int): Spaces per indent level.
        base_indent_level (int): Starting indent level from parent.
    Returns:
        str: Indented text block.
    """
    lines = text.splitlines()
    output_lines = []
    indent_unit = ' ' * indent_spaces

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line and not line.isspace():
            output_lines.append(indent_unit * base_indent_level)  # Empty line at base level
            continue

        # Apply base indentation only, no additional levels unless explicitly structured
        current_indent_prefix = indent_unit * base_indent_level
        output_lines.append(current_indent_prefix + stripped_line)

    return "\n".join(output_lines)

# --- Core Recursive Structure ---

DEBUG = False

def format_recursively(initial_data, indent_spaces=4):
    """
    Initiates recursive processing.
    """
    try:
        processed_data = _process_value(initial_data, indent_spaces, base_indent_level=0)
        return processed_data
    except Exception as e:
        raise

def _process_value(value, indent_spaces, base_indent_level, key=None):
    """
    Recursively processes values, adjusting indentation and escaping.
    Args:
        value: Value to process.
        indent_spaces: Spaces per indent level.
        base_indent_level: Current indent level from parent.
        key: The key associated with this value (e.g., "diff").
    """
    if isinstance(value, dict):
        return {k: _process_value(v, indent_spaces, base_indent_level + 1, k) for k, v in value.items()}
    elif isinstance(value, list):
        return [_process_value(item, indent_spaces, base_indent_level) for item in value]
    elif isinstance(value, str):
        if DEBUG: print(f"DEBUG: Processing string for key '{key}': '{value[:100]}...'")
        try:
            # Unescape the string fully
            unescaped_text = codecs.decode(value, 'unicode_escape')
            unescaped_text = unescaped_text.replace(r'\"', '"').replace(r'\\', '\\')
            if DEBUG: print(f"DEBUG: Unescaped text: '{unescaped_text[:100]}...'")

            # Apply indentation based on context
            if key == "diff":
                # For diffs, align with the "diff" key's indent level
                indented_text = simple_indent_text(unescaped_text, indent_spaces, base_indent_level)
            else:
                # For other strings, use minimal indentation unless nested content requires it
                indented_text = simple_indent_text(unescaped_text, indent_spaces, max(0, base_indent_level - 1))
            if DEBUG: print(f"DEBUG: Indented text: '{indented_text[:100]}...'")
            return indented_text
        except Exception as e:
            print(f"Warning: Failed to process string: {e}. String: '{value[:50]}...'", file=sys.stderr)
            return value
    else:
        return value

# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print JSON-like files with context-aware indentation and full unescaping."
    )
    parser.add_argument('glob_patterns', metavar='PATTERN', type=str, nargs='+', help='File path patterns.')
    parser.add_argument('--indent', type=int, default=4, help='Spaces for indentation (default: 4).')
    parser.add_argument('--debug', action='store_true', help='Enable debug printing.')

    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug
    if DEBUG: print("DEBUG: Debug mode enabled.", file=sys.stderr)

    has_processed_any_file = False
    exit_code = 0

    for pattern in args.glob_patterns:
        try:
            filepaths = glob.glob(os.path.expanduser(pattern), recursive=True)
        except Exception as e:
            print(f"Error: Could not expand glob pattern '{pattern}': {e}", file=sys.stderr)
            exit_code = 1
            continue

        if not filepaths:
            print(f"Warning: No files found matching pattern '{pattern}'", file=sys.stderr)
            continue

        has_processed_any_file = True

        for filepath in filepaths:
            if len(filepaths) > 1 or len(args.glob_patterns) > 1:
                print(f"\n--- File: {filepath} ---", file=sys.stderr)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                if not file_content.strip():
                    print(f"Warning: File '{filepath}' is empty.", file=sys.stderr)
                    continue

                data_object = json.loads(file_content, strict=False)
                processed_data = format_recursively(data_object, args.indent)
                final_output = json.dumps(processed_data, indent=args.indent, ensure_ascii=False)
                final_output = final_output.replace('\\n', '\n').replace('\\"', '"')
                print(final_output)

            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in '{filepath}': {e}", file=sys.stderr)
                exit_code = 1
            except Exception as e:
                print(f"Error processing '{filepath}': {e}", file=sys.stderr)
                exit_code = 1

    if not has_processed_any_file:
        print("No files found.", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
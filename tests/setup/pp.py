#!/usr/bin/env python3

import json
import codecs # For initial unescaping within strings
import argparse
import glob
import sys
import os
# import traceback # Keep commented out unless debugging specific formatting errors

# --- Simple Text Indenter ---

def simple_indent_text(text, indent_spaces=4):
    """
    Applies simple, bracket-based indentation to a block of text.
    Args:
        text (str): The text block to indent.
        indent_spaces (int): The number of spaces per indent level.
    Returns:
        str: The text block with indentation applied line by line.
    """
    lines = text.splitlines() # Split into individual lines
    output_lines = []
    current_indent_level = 0
    indent_unit = ' ' * indent_spaces

    for line in lines:
        # Preserve essential whitespace if line is just whitespace, otherwise strip
        if line.isspace():
            stripped_line = "" # Represent blank lines as empty
        else:
            stripped_line = line.strip() # Remove leading/trailing for analysis & printing

        # Skip fully empty lines (after stripping)
        if not stripped_line and not line.isspace(): # Handle lines that become empty after strip
             output_lines.append("")
             continue

        # --- Indentation Logic ---
        # 1. Check for closing tokens: Decrease indent level BEFORE processing this line
        if stripped_line.startswith(('}', ']', ')', '</')):
             current_indent_level = max(0, current_indent_level - 1) # Don't go below 0

        # 2. Apply current indentation
        current_indent_prefix = indent_unit * current_indent_level
        output_lines.append(current_indent_prefix + stripped_line)

        # 3. Check for opening tokens: Increase indent level AFTER processing this line (for the next line)
        if stripped_line.endswith(('{', '[', '(')):
             current_indent_level += 1
        # Basic check for opening XML/HTML tags (avoid self-closing/closing)
        elif stripped_line.startswith('<') and \
             not stripped_line.startswith('</') and \
             not stripped_line.endswith('/>'):
             current_indent_level += 1
        # --- End Indentation Logic ---

    return "\n".join(output_lines)


# --- Core Recursive Structure (Applies Simple Indenter to Strings) ---

DEBUG = False # Global debug flag, can be set by --debug argument

def format_recursively(initial_data):
    """
    Main function to initiate recursive processing.
    Expects a Python object. Returns a processed Python object where
    strings have been processed by the simple indenter.
    """
    try:
        processed_data = _process_value(initial_data)
        return processed_data
    except Exception as e:
        raise # Re-raise exceptions to be caught in main loop

def _process_value(value):
    """
    Recursively processes Python values. Applies simple indentation to strings.
    """
    if isinstance(value, dict):
        # Process dict values recursively
        return {k: _process_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Process list items recursively
        return [_process_value(item) for item in value]
    elif isinstance(value, str):
        # If it's a string, unescape basic sequences and apply simple indentation
        if DEBUG: print(f"DEBUG: Processing string for simple indent: '{value[:100]}...'")
        try:
            # Use unicode_escape for basic unescaping (\n, \t, \", \\ etc.)
            # This turns escaped sequences into their literal characters (like newlines)
            unescaped_text = codecs.decode(value, 'unicode_escape')
            if DEBUG: print(f"DEBUG: Unescaped text for indenting: '{unescaped_text[:100]}...'")

            # Apply the simple line-by-line indentation logic using the global indent setting
            # (We need access to args.indent here, or pass it down, or assume default)
            # Let's pass indent_spaces to simple_indent_text from main later.
            # For now, assume default 4 or modify simple_indent_text signature if needed.
            # --> Modification: Let's pass indent_spaces down for consistency.
            # --> We need args to be accessible or passed down. Let's pass it to format_recursively.

            # Apply the simple indentation logic
            # NOTE: Needs indent_spaces value passed down
            indented_text = simple_indent_text(unescaped_text) # Assumes default 4 for now
            if DEBUG: print(f"DEBUG: Result after simple indent: '{indented_text[:100]}...'")

            # Return the newly formatted string with literal newlines
            return indented_text
        except Exception as e:
            # If unescaping or indenting fails, return the original string to avoid crashing
            print(f"Warning: Failed to process/indent string: {e}. String: '{value[:50]}...'", file=sys.stderr)
            return value # Fallback to the original string
    else:
        # Numbers, booleans, None are returned as is
        return value

# --- Function for final newline replacement ---
def unescape_newlines_in_json_string(json_string):
    """
    Replaces escaped newline sequences '\\n' generated by json.dumps
    with literal newline characters '\n'.

    CAUTION: The output string is likely NOT valid JSON anymore.
    """
    # Replace the two-character sequence '\\n' with a literal newline '\n'
    return json_string.replace('\\n', '\n')

# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Recursively pretty-print JSON-like files, applying simple text indentation to all string values and rendering newlines.",
        epilog="Example: python pp.py data/*.json \"logs/**/*.log\" > output.txt"
    )
    parser.add_argument(
        'glob_patterns',
        metavar='PATTERN',
        type=str,
        nargs='+',
        help='One or more file path patterns (globs) to process (e.g., *.json, "project/**/*.json"). Quote patterns containing wildcards.'
    )
    parser.add_argument(
        '--indent',
        metavar='SPACES',
        type=int,
        default=4,
        help='Number of spaces for base structure indentation AND simple text indentation (default: 4).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug printing for string processing.'
    )

    args = parser.parse_args()

    # Set global DEBUG flag (or pass 'args.debug' down if preferred)
    global DEBUG
    if args.debug:
        DEBUG = True
        print("DEBUG: Debug mode enabled.", file=sys.stderr)

    # Modify _process_value to accept indent_spaces
    # This requires changing the function definition and recursive calls.
    # Let's adjust this part.

    # --- Updated Recursive Processing ---
    def format_recursively_with_indent(initial_data, indent_spaces):
        try:
            processed_data = _process_value_with_indent(initial_data, indent_spaces)
            return processed_data
        except Exception as e:
            raise

    def _process_value_with_indent(value, indent_spaces):
        if isinstance(value, dict):
            return {k: _process_value_with_indent(v, indent_spaces) for k, v in value.items()}
        elif isinstance(value, list):
            return [_process_value_with_indent(item, indent_spaces) for item in value]
        elif isinstance(value, str):
            if DEBUG: print(f"DEBUG: Processing string for simple indent: '{value[:100]}...'")
            try:
                unescaped_text = codecs.decode(value, 'unicode_escape')
                # Pass the correct indent_spaces to the simple indenter
                indented_text = simple_indent_text(unescaped_text, indent_spaces=indent_spaces)
                return indented_text
            except Exception as e:
                print(f"Warning: Failed to process/indent string: {e}. String: '{value[:50]}...'", file=sys.stderr)
                return value
        else:
            return value
    # --- End Updated Recursive Processing ---


    has_processed_any_file = False
    exit_code = 0

    for pattern in args.glob_patterns:
        try:
            expanded_pattern = os.path.expanduser(pattern)
            filepaths = glob.glob(expanded_pattern, recursive=True)
        except Exception as e:
            print(f"Error: Could not expand glob pattern '{pattern}': {e}", file=sys.stderr)
            exit_code = 1
            continue

        if not filepaths:
            if any(c in pattern for c in '*?[]'):
                 print(f"Warning: No files found matching pattern '{pattern}'", file=sys.stderr)
            elif not os.path.exists(expanded_pattern):
                 print(f"Error: File not found '{expanded_pattern}'", file=sys.stderr)
                 exit_code = 1
            continue

        has_processed_any_file = True

        for filepath in filepaths:
            # Print file separator to stderr
            is_multiple_files = len(filepaths) > 1 or len(args.glob_patterns) > 1
            if is_multiple_files:
                 print(f"\n--- File: {filepath} ---", file=sys.stderr)

            try:
                # Read file
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                if not file_content.strip():
                    print(f"Warning: File '{filepath}' is empty, skipping.", file=sys.stderr)
                    continue

                # Parse outer structure
                try:
                    data_object = json.loads(file_content, strict=False)
                except json.JSONDecodeError as e:
                    print(f"Error: Could not parse outer JSON structure in '{filepath}'. Invalid JSON: {e}", file=sys.stderr)
                    exit_code = 1
                    continue

                # Process recursively (applies simple_indent_text to strings)
                try:
                    # Pass the indent value from args
                    processed_data = format_recursively_with_indent(data_object, args.indent)

                    # Generate intermediate JSON string (escapes internal newlines)
                    intermediate_json_string = json.dumps(
                        processed_data,
                        indent=args.indent,
                        ensure_ascii=False
                    )

                    # **** Post-process to replace '\\n' with '\n' ****
                    final_output = unescape_newlines_in_json_string(intermediate_json_string)

                    # Print the final, potentially non-JSON, output to stdout
                    print(final_output)

                except Exception as e:
                    print(f"Error: Failed during recursive processing/formatting of '{filepath}': {e}", file=sys.stderr)
                    # if DEBUG: print(traceback.format_exc()) # Uncomment for debugging trace
                    exit_code = 1
                    continue

            # Handle file reading errors etc.
            except IOError as e:
                print(f"Error: Could not read file '{filepath}': {e}", file=sys.stderr)
                exit_code = 1
            except Exception as e:
                print(f"Error: An unexpected error occurred while processing '{filepath}': {e}", file=sys.stderr)
                # if DEBUG: print(traceback.format_exc()) # Uncomment for debugging trace
                exit_code = 1

    # Final status message if needed
    if not has_processed_any_file and exit_code == 0:
         print("No files matching the provided patterns were found.", file=sys.stderr)
         exit_code = 1

    sys.exit(exit_code)

# --- Boilerplate execution ---
if __name__ == "__main__":
    main()
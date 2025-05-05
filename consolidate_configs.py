import os
import shutil
import datetime
from pathlib import Path

# --- Configuration ---
PROJECT_BASE_PATH = Path("/Users/jim/src/apps/c4h_ai_dev") # Absolute path to your project

# Old config directories (relative to PROJECT_BASE_PATH)
OLD_CONFIG_DIR = "config"
OLD_CONFIG_TEAMS_DIR = "config_teams_0502"

# New consolidated config directory (relative to PROJECT_BASE_PATH)
NEW_CONFIG_DIR = "config" # Will overwrite OLD_CONFIG_DIR after backup

# Backup directory base name (will have timestamp appended)
BACKUP_BASE_NAME = "config_backup"

# Files to copy to the new structure
SYSTEM_CONFIG_SOURCE = Path(OLD_CONFIG_TEAMS_DIR) / "system_config.yml" # Source from the teams config backup
SCHEMA_FILES = ["system.json", "persona.json", "job.json"] # Schemas to copy from old 'config/schemas' backup
PERSONA_FILES_TEAMS = [ # Personas to copy from 'config_teams_0502/personas' backup
    "discovery_v1.yml",
    "solution_designer_v1.yml",
    "coder_v1.yml",
    "semantic_merge_v1.yml",
]
# Add any other essential personas from the old 'config/personas' if needed
# PERSONA_FILES_OLD = ["semantic_extract_v1.yml", ...]

# --- Helper Functions ---
def move_to_backup(source_path: Path, backup_dir: Path):
    """Moves a directory to the backup location if it exists."""
    if source_path.exists() and source_path.is_dir():
        try:
            destination = backup_dir / source_path.name
            shutil.move(str(source_path), str(destination))
            print(f"INFO: Moved '{source_path.name}' to '{destination}'")
        except Exception as e:
            print(f"ERROR: Failed to move '{source_path.name}' to backup: {e}")
    else:
        print(f"INFO: Directory '{source_path.name}' not found, skipping backup.")

def copy_file(source_path: Path, dest_path: Path):
    """Copies a single file, creating destination directory if needed."""
    if source_path.exists() and source_path.is_file():
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(dest_path)) # copy2 preserves metadata
            print(f"INFO: Copied '{source_path}' to '{dest_path}'")
        except Exception as e:
            print(f"ERROR: Failed to copy '{source_path}' to '{dest_path}': {e}")
    else:
        print(f"WARNING: Source file '{source_path}' not found, skipping copy.")

# --- Main Script Logic ---
def main():
    print("--- Starting Configuration Consolidation ---")

    # Define absolute paths
    old_config_path = PROJECT_BASE_PATH / OLD_CONFIG_DIR
    old_config_teams_path = PROJECT_BASE_PATH / OLD_CONFIG_TEAMS_DIR
    new_config_path = PROJECT_BASE_PATH / NEW_CONFIG_DIR

    # 1. Create Backup Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir_name = f"{BACKUP_BASE_NAME}_{timestamp}"
    backup_path = PROJECT_BASE_PATH / backup_dir_name
    try:
        backup_path.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Created backup directory: '{backup_path}'")
    except Exception as e:
        print(f"ERROR: Could not create backup directory '{backup_path}': {e}")
        return # Stop if backup dir fails

    # 2. Backup Old Directories
    print("\n--- Backing up old configuration directories ---")
    move_to_backup(old_config_path, backup_path)
    move_to_backup(old_config_teams_path, backup_path)

    # Define paths relative to the backup directory for sourcing files
    backup_old_config_path = backup_path / OLD_CONFIG_DIR
    backup_teams_config_path = backup_path / OLD_CONFIG_TEAMS_DIR

    # 3. Create New Structure
    print("\n--- Creating new configuration structure ---")
    new_schemas_path = new_config_path / "schemas"
    new_personas_path = new_config_path / "personas"
    try:
        new_config_path.mkdir(exist_ok=True)
        new_schemas_path.mkdir(exist_ok=True)
        new_personas_path.mkdir(exist_ok=True)
        print(f"INFO: Created directory: '{new_config_path}'")
        print(f"INFO: Created directory: '{new_schemas_path}'")
        print(f"INFO: Created directory: '{new_personas_path}'")
    except Exception as e:
        print(f"ERROR: Failed to create new directory structure: {e}")
        return

    # 4. Populate New Config Directory
    print("\n--- Populating new configuration directory ---")

    # Copy system_config.yml
    system_config_source_abs = backup_teams_config_path / SYSTEM_CONFIG_SOURCE.name # Get name only
    copy_file(system_config_source_abs, new_config_path / "system_config.yml")

    # Copy Schemas
    print(f"\n--- Copying Schemas from '{backup_old_config_path / 'schemas'}' ---")
    for schema_file in SCHEMA_FILES:
        schema_source_abs = backup_old_config_path / "schemas" / schema_file
        copy_file(schema_source_abs, new_schemas_path / schema_file)

    # Copy Personas from config_teams_0502 backup
    print(f"\n--- Copying Personas from '{backup_teams_config_path / 'personas'}' ---")
    for persona_file in PERSONA_FILES_TEAMS:
        persona_source_abs = backup_teams_config_path / "personas" / persona_file
        copy_file(persona_source_abs, new_personas_path / persona_file)

    # Copy any additional essential personas from the old config backup if defined
    # if 'PERSONA_FILES_OLD' in locals() and PERSONA_FILES_OLD:
    #     print(f"\n--- Copying Additional Personas from '{backup_old_config_path / 'personas'}' ---")
    #     for persona_file in PERSONA_FILES_OLD:
    #         persona_source_abs = backup_old_config_path / "personas" / persona_file
    #         # Avoid overwriting if it already exists from the teams backup
    #         if not (new_personas_path / persona_file).exists():
    #             copy_file(persona_source_abs, new_personas_path / persona_file)
    #         else:
    #             print(f"INFO: Persona '{persona_file}' already copied from teams backup, skipping.")

    print("\n--- Configuration Consolidation Complete ---")
    print(f"Old configurations backed up to: '{backup_path}'")
    print(f"New consolidated configuration created at: '{new_config_path}'")

if __name__ == "__main__":
    main()
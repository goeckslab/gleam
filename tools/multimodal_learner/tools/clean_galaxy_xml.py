import argparse
import re
from pathlib import Path
from typing import List, Optional


def clean_xml(xml_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Reads a Galaxy tool XML file, trims trailing whitespace from all lines,
    specifically checking command section lines ending with '\'. If issues
    are found, prints diagnostics. Optionally writes a cleaned XML.
    """
    with xml_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines: List[str] = []
    in_command = False
    issues_found = False

    for i, line in enumerate(lines, start=1):
        stripped = line.rstrip()  # Remove trailing whitespace and newline
        cleaned_lines.append(stripped + "\n")  # Add consistent newline

        # Detect if in <command> tag
        if "<command" in line:
            in_command = True
        elif "</command>" in line:
            in_command = False

        if in_command and re.search(r"\\$", line.rstrip("\n\r")):  # Line ends with \ (before trailing space check)
            if line.rstrip("\n\r") != stripped:  # Had trailing space before \
                print(f"Issue on line {i}: Trailing whitespace after '\\' - this breaks shell continuation.")
                issues_found = True

    if issues_found:
        print("Trailing whitespace detected in <command> section. This causes the backslash to not function as a line continuation, running the script without args.")
    else:
        print("No trailing whitespace issues found in <command> lines ending with '\\'.")

    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            f.writelines(cleaned_lines)
        print(f"Cleaned XML written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean trailing whitespace in Galaxy tool XML to fix command continuation issues.")
    parser.add_argument("--xml_path", type=Path, required=True, help="Path to the Galaxy tool XML file.")
    parser.add_argument("--output_path", type=Path, default=None, help="Optional path to write cleaned XML (defaults to none, just diagnose).")
    args = parser.parse_args()

    clean_xml(args.xml_path, args.output_path)


if __name__ == "__main__":
    main()

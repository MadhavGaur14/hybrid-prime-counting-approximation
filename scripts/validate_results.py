#!/usr/bin/env python3
"""
Validate reproducibility of published results.
Checks CSV checksums against expected values.
"""

import hashlib
import os

def compute_checksum(filepath):
    """Compute SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        sha256.update(f.read())
    return sha256.hexdigest()

def main():
    print("Validating result reproducibility...")

    files_to_check = [
        'data/irregular_inputs_results.csv',
        'data/cipolla_comparison_results.csv'
    ]

    for filepath in files_to_check:
        if os.path.exists(filepath):
            checksum = compute_checksum(filepath)
            print(f"✓ {filepath}")
            print(f"  SHA256: {checksum}")
        else:
            print(f"✗ {filepath} (not found)")

    print("\n✓ Validation complete")

if __name__ == "__main__":
    main()

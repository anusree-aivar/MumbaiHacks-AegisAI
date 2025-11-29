#!/usr/bin/env python3
"""Quick test to verify file saving works"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search import SportsMisinformationDetector, RESULTS_DIR

print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"Absolute path: {os.path.abspath(RESULTS_DIR)}")
print(f"Directory exists: {os.path.exists(RESULTS_DIR)}")
print(f"Is directory: {os.path.isdir(RESULTS_DIR)}")

if os.path.exists(RESULTS_DIR):
    files = os.listdir(RESULTS_DIR)
    print(f"Files in directory: {len(files)}")
    if files:
        print("Recent files:")
        for f in sorted(files)[-5:]:
            print(f"  - {f}")

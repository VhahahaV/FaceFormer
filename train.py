#!/usr/bin/env python
"""Main training script for FaceFormer."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from main.train import main

if __name__ == '__main__':
    main()


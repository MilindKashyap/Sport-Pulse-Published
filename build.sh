#!/bin/bash
set -e

# Upgrade pip first
python -m pip install --upgrade pip

# Install setuptools and wheel explicitly
python -m pip install --upgrade setuptools wheel

# Install all requirements
python -m pip install -r requirements.txt


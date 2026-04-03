#!/usr/bin/env python3
"""
Train a collision prediction model for the spacecraft debris simulation

Usage:
    python train_model.py
"""

from ml_model import train_and_save_model
import sys

def main():
    print("=" * 60)
    print("Training Collision Prediction Model")
    print("=" * 60)
    print()
    
    try:
        model_path = train_and_save_model()
        print(f"\n{'='*60}")
        print("SUCCESS! Model trained and saved.")
        print(f"Model location: {model_path}")
        print("\nThe simulation will automatically use this model")
        print("when ML predictor mode is selected.")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

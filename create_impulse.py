"""
Impulse Creation Script for IoT ML

This script helps you define and create an "impulse" (data processing pipeline) for your ML model, supporting image, audio, and numerical/character data. The impulse includes preprocessing, feature extraction, and model input configuration, similar to Edge Impulse's workflow.

Usage:
    - Run the script and follow the prompts to select your data type (image, audio, numerical, or character).
    - The script will guide you through setting up preprocessing and feature extraction steps for your data type.
    - Outputs a summary of the impulse configuration.

Note: Extend this script to integrate with your dataset and downstream ML pipeline.
"""

import json

from ml_utils import read_choice


def choose_data_type():
    types = ["image", "audio", "numerical", "character"]
    print("Available data types:")
    return read_choice("Select your data type (number): ", types)

def configure_impulse(data_type):
    impulse = {"data_type": data_type, "preprocessing": [], "feature_extraction": [], "input_shape": None}
    if data_type == "image":
        impulse["preprocessing"] = ["resize", "normalize"]
        impulse["feature_extraction"] = ["flatten", "edge_detection (optional)"]
        impulse["input_shape"] = input("Enter image input shape (e.g., 64x64x1): ")
    elif data_type == "audio":
        impulse["preprocessing"] = ["resample", "normalize"]
        impulse["feature_extraction"] = ["MFCC", "spectrogram"]
        impulse["input_shape"] = input("Enter audio input shape (e.g., 16000 samples): ")
    elif data_type == "numerical":
        impulse["preprocessing"] = ["normalize", "impute_missing"]
        impulse["feature_extraction"] = ["statistical_features", "fft (optional)"]
        impulse["input_shape"] = input("Enter number of features: ")
    elif data_type == "character":
        impulse["preprocessing"] = ["tokenize", "pad_sequences"]
        impulse["feature_extraction"] = ["embedding", "n-grams (optional)"]
        impulse["input_shape"] = input("Enter max sequence length: ")
    return impulse

def main():
    data_type = choose_data_type()
    impulse = configure_impulse(data_type)
    # --- Learning Block (Simple Classifier) ---
    print("\n--- Learning Block ---")
    print("Use this configuration with generate_features.py and train_model.py:")
    print("1. Load your dataset and apply the above preprocessing and feature extraction.")
    print("2. Train a classifier (e.g., SVM, RandomForest, or a neural network) on the features.")
    print("3. Evaluate and export the model for your IoT device.")

    models = ["SVM", "RandomForest", "NeuralNetwork"]
    print("Available classifiers:")
    chosen_model = read_choice("Select a classifier for this impulse (number): ", models)
    impulse["classifier"] = chosen_model

    print("\nImpulse configuration:")
    print(json.dumps(impulse, indent=2))

    save = input("Save impulse configuration to file? (y/n): ").strip().lower()
    if save == "y":
        fname = input("Enter filename (e.g., impulse_config.json): ")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(impulse, f, indent=2)
        print(f"Impulse configuration saved to {fname}")
    print(f"\nConfigured classifier: {chosen_model}")

if __name__ == "__main__":
    main()

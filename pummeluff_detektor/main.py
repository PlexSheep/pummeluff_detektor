import argparse
import numpy as np
from PIL import Image
import sys
from pathlib import Path

import loader


def process_image(image_path: str, target_size=(64, 64)) -> np.ndarray:
    """
    Load and preprocess a single image for classification.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing

    Returns:
        Preprocessed image array
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # Apply the same preprocessing as during training
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, -1)  # Add batch dimension and flatten

        return img_array

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        sys.exit(1)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Detect if an image is a Jigglypuff')
    parser.add_argument('image_path', type=str,
                        help='Path to the image file to classify')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for Jigglypuff detection (default: 0.5)')

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Load the model
    print("Loading model...")
    model, label_encoder, metrics = loader.load()

    print("\nAvailable classes:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"{idx}: {class_name}")

    # Find Jigglypuff class index (case-insensitive)
    try:
        classes_lower = [c.lower() for c in label_encoder.classes_]
        jigglypuff_idx = classes_lower.index('jigglypuff')
    except ValueError:
        print("Error: Model was not trained with Jigglypuff class!")
        sys.exit(1)

    # Process the image
    print("Processing image...")
    img_array = process_image(str(image_path))

    # Get prediction probabilities
    pred_probs = model.predict_proba(img_array)[0]
    jigglypuff_prob = pred_probs[jigglypuff_idx]

    # Get predicted class
    pred_class = label_encoder.classes_[pred_probs.argmax()]

    # Print results
    print("\nResults:")
    print(f"Image: {image_path}")
    print(f"Jigglypuff probability: {jigglypuff_prob:.2%}")
    print(f"Predicted class: {pred_class}")

    if jigglypuff_prob >= args.threshold:
        print("\nVerdict: This appears to be a Jigglypuff! ꒰◍ˊ◡ˋ꒱")
    else:
        print("\nVerdict: This does not appear to be a Jigglypuff (｡•́︿•̀｡)")


if __name__ == "__main__":
    main()

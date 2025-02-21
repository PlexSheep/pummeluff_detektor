import sys
import numpy as np

EM_HAPPY = '✧٩(•́⌄•́๑)و ✧'
EM_SAD = '(╥﹏╥)'
EM_UNSURE = r'¯\_(Φ ᆺ Φ)_/¯'


def process_image(image_path: str, target_size=(64, 64)) -> np.ndarray:
    """
    Load and preprocess a single image for classification.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing

    Returns:
        Preprocessed image array
    """
    from PIL import Image

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


def detector(args):
    import loader
    from pathlib import Path

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Oh no {EM_SAD}: the image file does not exist {EM_UNSURE}")
        sys.exit(1)

    # Load the model
    print("Loading model...")
    detector = loader.Detector.load_or_train(
        verbose=args.verbose, force_training=args.train)
    if detector is None:
        print(f"Oh no {EM_SAD}: the detector could not be loaded")
        sys.exit(1)

    if args.verbose:
        print(f"\n{detector.info()}")

    # Find Jigglypuff class index (case-insensitive)
    try:
        classes_lower = [c.lower() for c in detector.label_encoder.classes_]
        jigglypuff_idx = classes_lower.index('jigglypuff')
    except ValueError:
        print("Error: Model was not trained with Jigglypuff class!")
        sys.exit(1)

    # Process the image
    print("Processing image...")
    img_array = process_image(str(image_path))

    # Get prediction probabilities
    pred_probs = detector.model.predict_proba(img_array)[0]
    jigglypuff_prob = pred_probs[jigglypuff_idx]

    # Get predicted class
    pred_class = detector.label_encoder.classes_[pred_probs.argmax()]

    # Print results
    print("\nResults:")
    print(f"Image: {image_path}")
    print(f"Jigglypuff probability: {jigglypuff_prob:.2%}")
    print(f"Predicted class: {pred_class}")

    if pred_class == "Jigglypuff":
        print(
            f"\nVerdict: This image appears to contain a Jigglypuff! {EM_HAPPY}")
        if jigglypuff_prob < args.threshold:
            print(f"\t but not so sure {EM_UNSURE}")
    else:
        print(
            f"\nVerdict: This image does not appear to contain a Jigglypuff {EM_SAD}")


def main():
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Detect if an image contains a Jigglypuff')
    parser.add_argument('image_path', type=str,
                        help='Path to the image file to classify')
    parser.add_argument('--train', action="store_true",
                        help='Train the model and save to file')
    parser.add_argument('--verbose', '-v', action="store_true",
                        help='Show many output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for Jigglypuff detection (default: 0.5)')

    args = parser.parse_args()

    try:
        print("Loading detector...")
        detector(args)
    except KeyboardInterrupt:
        print(f"{EM_SAD} I was interrupted {EM_SAD}")


if __name__ == "__main__":
    main()

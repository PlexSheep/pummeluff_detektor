import argparse
import sys
import numpy as np
from pathlib import Path
import logging

from pummeluff_detektor.loader import Detector

EM_HAPPY = '✧٩(•́⌄•́๑)و ✧'
EM_SAD = '(╥﹏╥)'
EM_UNSURE = r'¯\_(Φ ᆺ Φ)_/¯'
EM_ANGRY = 'ヽ(｀⌒´メ)ノ'


def setup_logging(verbose: bool) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose (bool): If True, sets DEBUG level and adds file logging.
                       If False, sets INFO level with console only.

    Note:
        Creates 'pummeluff_detector.log' file when verbose is True
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)

    # Optional file handler for debug logging
    if verbose:
        file_handler = logging.FileHandler('pummeluff_detector.log')
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    if verbose:
        root_logger.addHandler(file_handler)


def process_image(image_path: str, target_size=(64, 64)) -> np.ndarray:
    """
    Load and preprocess a single image for classification.

    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing, default (64, 64)

    Returns:
        np.ndarray: Preprocessed image array (1, height * width * 3)

    Raises:
        SystemExit: If image processing fails
    """
    from PIL import Image
    logger = logging.getLogger(__name__)

    try:
        logger.debug(f"Processing image: {image_path}")
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


def load_detector(args) -> Detector:
    from pummeluff_detektor import loader
    logger = logging.getLogger(__name__)

    logger.info("loading detector...")
    detector = loader.Detector.load_or_train(
        force_training=args.train,
        training_images_dir=args.training_images
    )
    if detector is None:
        logger.error("Failed to load or train detector")
        print(f"Oh no {EM_SAD}: the detector could not be loaded")
        sys.exit(1)
    logger.info(f"done loading detector {EM_HAPPY}")
    return detector


def info(args: argparse.Namespace):
    """
    Print information about the model and detector, then exit
    """

    detector = load_detector(args)

    print(f"\n{detector.info()}")


def detector(args: argparse.Namespace):
    """
    Main detection function that processes an image and reports results.

    Args:
        args (argparse.Namespace): Command line arguments including:
            - image_path: Path to image to classify
            - verbose: Whether to show debug output
            - train: Whether to force training new model
            - threshold: Confidence threshold for detection
            - training_images: Optional path to training images

    Raises:
        SystemExit: On various error conditions
    """
    logger = logging.getLogger(__name__)
    from pummeluff_detektor import loader

    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        print(f"Oh no {EM_SAD}: the image file does not exist {EM_UNSURE}")
        sys.exit(1)

    detector = load_detector(args)

    logger.debug(f"\n{detector.info()}")

    # Find Jigglypuff and Kirby class indices
    classes_lower = [c.lower() for c in detector.label_encoder.classes_]
    try:
        jigglypuff_idx = classes_lower.index('jigglypuff')
        kirby_idx = classes_lower.index('kirby')
    except ValueError as e:
        logger.error(f"Required class not found in model: {e}")
        sys.exit(1)

    logger.info("Processing image...")
    img_array = process_image(str(image_path))

    # Get prediction probabilities
    pred_probs = detector.model.predict_proba(img_array)[0]
    jigglypuff_prob = pred_probs[jigglypuff_idx]
    kirby_prob = pred_probs[kirby_idx]
    pred_class = str(detector.label_encoder.classes_[
        pred_probs.argmax()]).lower()

    logger.debug(
        f"Prediction probabilities: Jigglypuff={jigglypuff_prob:.2%}, Kirby={kirby_prob:.2%}")

    # Print results
    print("\nResults:")
    print(f"Image: {image_path}")
    print(f"Jigglypuff probability: {jigglypuff_prob:.2%}")
    print(f"Kirby probability: {kirby_prob:.2%}")
    print(f"Predicted class: {pred_class}")

    if pred_class == "jigglypuff":
        msg = f"Verdict: This image appears to contain a Jigglypuff! {EM_HAPPY}"
        if jigglypuff_prob < args.threshold:
            msg += f"\n\t but not so sure {EM_UNSURE}"
        print(msg)
        logger.info(
            f"Detected Jigglypuff with {jigglypuff_prob:.2%} confidence")
    else:
        msg = f"Verdict: This image does not appear to contain a Jigglypuff {EM_SAD}"
        if pred_class == "kirby":
            msg += f"\n\t BUT IT SEEMS TO CONTAIN A KIRBY!! {EM_ANGRY}"
        print(msg)
        logger.info(
            f"Detected {pred_class} with {pred_probs.max():.2%} confidence")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect if an image contains a Jigglypuff')
    parser.add_argument('image_path', type=str,
                        help='Path to the image file to classify', nargs='?')
    parser.add_argument('--train', action="store_true",
                        help='Train the model and save to file')
    parser.add_argument('--verbose', '-v', action="store_true",
                        help='Show debug output')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for Jigglypuff detection (default: 0.5)')
    parser.add_argument('--training-images', default=None,
                        help='Path to directory with custom training images')
    parser.add_argument('-i', "--info", action="store_true",
                        help="show info and exit")

    args = parser.parse_args()

    # Set up logging before anything else
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.training_images is not None:
        args.training_images = Path(args.training_images)

    try:
        if args.info:
            info(args)
        else:
            if args.image_path is None:
                parser.print_usage()
                parser.exit(1)
            logger.info("Starting Pummeluff detector...")
            detector(args)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        print(f"{EM_SAD} I was interrupted {EM_SAD}")


if __name__ == "__main__":
    main()

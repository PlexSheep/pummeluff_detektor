from __future__ import annotations
import logging
import kagglehub
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm
import joblib
from pathlib import Path
from typing import Tuple, List, Optional,  Any
import appdirs

# Custom types
ImageArray = np.ndarray  # Shape: (height, width, 3)
BatchImageArray = np.ndarray  # Shape: (n_samples, height, width, 3)
FlatImageArray = np.ndarray  # Shape: (n_samples, height * width * 3)
Labels = np.ndarray  # Shape: (n_samples,)
ProcessArgs = Tuple[str, str, Tuple[int, int]]

SEED: int = 19

DATA_DIR = Path(appdirs.user_data_dir("pummeluff-detektor"))
MODEL_DIR = DATA_DIR / "models"

# For downloaded datasets
CACHE_DIR = Path(appdirs.user_cache_dir("pummeluff-detektor"))
DATASET_DIR = CACHE_DIR / "datasets"


class Detector:
    """
    A detector class for identifying something in images, specifically trained to distinguish
    Jigglypuff from other motives (especially kirby).

    Attributes:
        model (RandomForestClassifier): The trained classifier model
        label_encoder (LabelEncoder): Encoder for converting between label names and indices
        class_report (dict): Classification report containing precision, recall, etc.
        feature_importance (Any): Feature importance scores from the model
        seed (int): seed for randomized things, for reproducibility (default is the SEED constant)
    """
    model: RandomForestClassifier
    label_encoder: LabelEncoder
    class_report: dict[Any, Any]
    feature_importance: Any
    seed = SEED

    def __init__(
        self,
        model: RandomForestClassifier,
        label_encoder: LabelEncoder,
        class_report: dict[Any, Any]
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.label_encoder = label_encoder
        self.class_report = class_report

        # Create directories if they don't exist
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    def get_class_labels(self) -> list[str]:
        """
        Get list of class labels that the model can detect.

        Returns:
            list[str]: List of class names as strings
        """
        return [str(c) for c in self.label_encoder.classes_]

    def info(self) -> str:
        """
        Get a formatted string containing model information.

        Returns:
            str: Multi-line string containing model type, parameters, available classes,
                and classification report
        """
        buf = ""
        buf += f"Model type: {str(self.model.__class__.__name__)}\n\n"
        buf += f"Parameters:\n"

        for (k, v) in self.model.get_params().items():
            buf += f"{k:<26}: {v}\n"
        buf += "\nAvailable classes:\n"
        for idx, class_name in enumerate(self.get_class_labels()):
            buf += f"{idx:02}: {class_name}\n"

        buf += f"Class report\n{self.class_report}"

        return buf

    def save(self, base_path: Path = MODEL_DIR) -> None:
        """
        Save the detector to a joblib file.

        Args:
            base_path (Path): Directory where the model should be saved
                            (default: MODEL_DIR)
        """
        try:
            os.mkdir(base_path)
        except FileExistsError:
            pass
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detector_path = os.path.join(
            base_path, f'pummeluff_detektor_{timestamp}.joblib')
        self.logger.info(f"Saving model to {detector_path}")
        joblib.dump(self, detector_path)

    @staticmethod
    def load(detector_path: Path) -> Detector:
        """
        Load a detector from a joblib file.

        Args:
            detector_path (Path): Path to the saved detector file

        Returns:
            Detector: Loaded detector instance

        Raises:
            FileNotFoundError: If detector file doesn't exist
            ValueError: If file is not a valid detector
        """
        logger = logging.getLogger(__name__)
        logger.info(f"trying to load a detector at: {detector_path}")
        d: Detector = joblib.load(detector_path)
        return d

    @staticmethod
    def train(training_images_dir: Path | None = None) -> Detector:
        """
        Train a new detector model.

        Args:
            training_images_dir (Path | None): Optional path to custom training images.
                                               If None, downloads standard dataset.

        Returns:
            Detector: Trained detector instance

        Note:
            Saves confusion matrix plot to 'confusion.png'
            Automatically saves the trained model
        """
        logger = logging.getLogger(__name__)
        # Set random seed for reproducibility
        np.random.seed(SEED)

        # Load the dataset using parallel processing

        logger.info("Loading Image dataset...")
        if training_images_dir is None:
            data_base_path: Path = Detector.download_dataset()
        else:
            logger.info(
                f"Using custom training image directory: {training_images_dir}")
            data_base_path = training_images_dir

        X, y, label_encoder = Detector.load_image_dataset(
            data_base_path, target_size=(64, 64))

        logger.debug("\nDataset Summary:")
        logger.debug(f"Number of images: {len(X)}")
        logger.debug(f"Image shape: {X[0].shape}")
        logger.debug("\nClass distribution:")
        for class_name, count in zip(label_encoder.classes_, np.bincount(y)):
            logger.debug(f"{class_name}: {count} images")

        X = X.astype('float32') / 255.0
        X = X.reshape(X.shape[0], -1)

        random_state: int = np.random.randint(1_000_000_000)
        logger.debug(
            f"Using random state: {random_state} for train/test split")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        logger.info("Training Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=25,
            n_jobs=-1,
            random_state=SEED
        )

        rf_classifier.fit(X_train, y_train)
        logger.info("Training completed")

        y_pred = rf_classifier.predict(X_test)
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        logger.debug(f"Classification report:\n{class_report}")

        d = Detector(rf_classifier, label_encoder,
                     class_report=class_report)
        d.save()

        plot_confusion_matrix(
            y_true=y_test, y_pred=y_pred,
            classes=d.get_class_labels(),
            title="Pummeluff Detektor Confusion Matrix")

        return d

    @staticmethod
    def load_or_train(
            training_images_dir: Path | None = None,
            force_training: bool = False,
            base_path: Path = MODEL_DIR
    ) -> Detector:
        """
        Load the most recent model if it exists, otherwise train a new one.

        Args:
            training_images_dir (Path | None): Optional path to custom training images
            force_training (bool): If True, always train new model regardless of existing ones
            base_path (Path): Directory to look for existing models

        Returns:
            Detector: Loaded or newly trained detector instance
        """
        logger = logging.getLogger(__name__)
        if not force_training:
            logger.info("Trying to load the latest model...")
            loaded = Detector.load_latest(base_path)

            if loaded is not None:
                try:
                    logger.info("Successfully loaded existing model!")
                except FileNotFoundError as e:
                    logger.warning(f"Error loading existing model: {e}")
                    logger.info("Will train a new model instead.")
            else:
                logger.error("No existing model found. Will train a new one.")
        return Detector.train(training_images_dir=training_images_dir)

    @staticmethod
    def load_latest(base_path: Path = MODEL_DIR) -> Detector | None:
        """
        Load the most recently saved detector model.

        Args:
            base_path (Path): Directory to search for model files

        Returns:
            Detector | None: Most recent detector if found, None otherwise
        """
        p = Detector.get_latest_detector_path(base_path)
        if p is not None:
            return Detector.load(p)
        else:
            return None

    @staticmethod
    def get_latest_detector_path(base_path: Path = MODEL_DIR) -> Path | None:
        """
        Get the path to the most recently saved detector model.

        Args:
            base_path (Path): Directory to search for model files

        Returns:
            Path | None: Path to most recent detector file if found, None otherwise
        """
        if not os.path.exists(base_path):
            return None

        # Find all joblib files
        detector_files = []

        for f in os.listdir(base_path):
            if not f.endswith('.joblib'):
                continue

            full_path = os.path.join(base_path, f)
            if f.startswith('pummeluff_detektor_'):
                detector_files.append(full_path)

        # Make sure we have at least one complete set
        if not (detector_files):
            return None

        # Get the latest files by modification time
        return max(detector_files, key=os.path.getmtime)

    @staticmethod
    def download_dataset() -> Path:
        """
        Download the default dataset from Kaggle.

        Returns:
            Path: Path to the downloaded dataset directory
        """
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        path = kagglehub.dataset_download(
            "plexsheep/jigglypuff-detection-data")

        return Path(path)

    @staticmethod
    def load_image_dataset(
        base_path: Path,
        target_size: Tuple[int, int],
        n_jobs: Optional[int] = None
    ) -> Tuple[BatchImageArray, Labels, LabelEncoder]:
        """
        Load training images and their labels using parallel processing.

        Args:
            base_path (Path): Root directory containing class subdirectories
            target_size (Tuple[int, int]): Size to resize images to (width, height)
            n_jobs (Optional[int]): Number of parallel processes to use.
                                  If None, uses all CPU cores.

        Returns:
            Tuple containing:
                BatchImageArray: Array of processed images (n_samples, height, width, 3)
                Labels: Array of numeric labels
                LabelEncoder: Encoder for converting between label names and indices
        """
        if n_jobs is None:
            n_jobs = cpu_count()

        # Get all Pokemon classes
        classes: List[str] = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]

        # Prepare arguments for parallel processing
        all_image_args: List[ProcessArgs] = []
        for tclass in classes:
            pokemon_path = os.path.join(base_path, tclass, tclass)

            if not os.path.exists(pokemon_path):
                continue

            valid_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [
                f for f in os.listdir(pokemon_path)
                if os.path.splitext(f.lower())[1] in valid_extensions
            ]

            for img_file in image_files:
                img_path = os.path.join(pokemon_path, img_file)
                all_image_args.append((img_path, tclass, target_size))

        # Process images in parallel
        print(
            f"\nProcessing {len(all_image_args)} images using {n_jobs} processes...")
        with Pool(n_jobs) as pool:
            results = list(tqdm(
                pool.imap(load_and_process_image, all_image_args),
                total=len(all_image_args),
                desc="Loading images"
            ))

        # Filter out None results and separate images and labels
        results = [r for r in results if r is not None]
        images, labels = zip(*results)

        X: BatchImageArray = np.array(images)

        # Encode labels
        label_encoder = LabelEncoder()
        y: Labels = label_encoder.fit_transform(labels)

        return X, y, label_encoder


def load_and_process_image(args: ProcessArgs) -> Optional[Tuple[ImageArray, str]]:
    """
    Load and process a single image.

    Args:
        args: Tuple containing (image_path, pokemon_name, target_size)

    Returns:
        Tuple of (image_array, pokemon_name) if successful, None if failed
    """
    img_path, pokemon, target_size = args
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img)
        return img_array, pokemon
    except Exception as e:
        print(f"Error loading {img_path}: {str(e)}")
        return None


def plot_confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    classes: List[str],
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot confusion matrix using seaborn's heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion.png")

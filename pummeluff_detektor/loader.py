from __future__ import annotations
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

# Custom types
ImageArray = np.ndarray  # Shape: (height, width, 3)
BatchImageArray = np.ndarray  # Shape: (n_samples, height, width, 3)
FlatImageArray = np.ndarray  # Shape: (n_samples, height * width * 3)
Labels = np.ndarray  # Shape: (n_samples,)
ProcessArgs = Tuple[str, str, Tuple[int, int]]

SEED: int = 19
STANDARD_BASE_PATH: Path = Path("models")


class Detector:
    model: RandomForestClassifier
    label_encoder: LabelEncoder
    class_report: dict[Any, Any]
    feature_importance: Any
    verbose: bool = False
    seed = SEED

    def __init__(self,
                 model: RandomForestClassifier,
                 label_encoder: LabelEncoder,
                 class_report: dict[Any, Any],
                 verbose: bool = False) -> None:
        self.model = model
        self.label_encoder = label_encoder
        self.verbose = verbose
        self.class_report = class_report

    def get_class_labels(self) -> list[str]:
        l = []
        for c in self.label_encoder.classes_:
            l.append(str(c))
        return l

    def info(self) -> str:
        buf = ""
        buf += f"Model type: {str(self.model.__class__.__name__)}\n\n"
        buf += f"Parameters:\n"

        for (k, v) in self.model.get_params().items():
            buf += f"{k:<26}: {v}\n"
        buf += "\nAvailable classes:\n"
        for idx, class_name in enumerate(self.get_class_labels()):
            buf += f"{idx:02}: {class_name}\n"

        print(f"Class report\n{self.class_report}")

        return buf

    def save(self, base_path: Path = STANDARD_BASE_PATH) -> None:
        try:
            os.mkdir(base_path)
        except FileExistsError:
            pass
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detector_path = os.path.join(
            base_path, f'pummeluff_detektor_{timestamp}.joblib')
        joblib.dump(self, detector_path)

    @staticmethod
    def load(detector_path: Path, verbose: bool = False) -> Detector:
        d: Detector = joblib.load(detector_path)
        d.verbose = verbose
        return d

    @staticmethod
    def train(verbose: bool = True) -> Detector:
        # If we get here, either no model exists or loading failed
        # Set random seed for reproducibility
        np.random.seed(SEED)

        # Load the dataset using parallel processing
        print("Loading Pokemon dataset...")
        data_base_path: Path = Detector.download_dataset()  # Get the dataset path
        X, y, label_encoder = Detector.load_image_dataset(
            data_base_path, target_size=(64, 64))

        if verbose:
            # Print dataset information
            print("\nDataset Summary:")
            print(f"Number of images: {len(X)}")
            print(f"Image shape: {X[0].shape}")
            print("\nClass distribution:")
            for class_name, count in zip(label_encoder.classes_, np.bincount(y)):
                print(f"{class_name}: {count} images")
            print("\n")

        # Preprocess the data
        X = X.astype('float32') / 255.0  # Normalize pixel values
        X = X.reshape(X.shape[0], -1)    # Flatten the images

        random_state: int = np.random.randint(1_000_000_000)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        print("\nTraining Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=25,
            n_jobs=-1,
            random_state=SEED
        )

        rf_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_classifier.predict(X_test)

        # Generate metrics
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        d = Detector(rf_classifier, label_encoder,
                     class_report=class_report, verbose=verbose)
        d.save()
        return d

    @staticmethod
    def load_or_train(verbose: bool = False, force_training: bool = False, base_path: Path = STANDARD_BASE_PATH) -> Detector:
        """
        Load the most recent model if it exists, otherwise train a new one.
        """
        if not force_training:
            if verbose:
                print("Trying to load the latest model...")
            loaded = Detector.load_latest(base_path)

            if loaded is not None:
                try:
                    loaded.verbose = verbose
                    if verbose:
                        print("Successfully loaded existing model!")
                    return loaded
                except FileNotFoundError as e:
                    print(f"Error loading existing model: {e}")
                    print("Will train a new model instead.")
            else:
                print("No existing model found. Will train a new one.")
        return Detector.train(verbose=verbose)

    @staticmethod
    def load_latest(base_path: Path = STANDARD_BASE_PATH, verbose: bool = False) -> Detector | None:
        p = Detector.get_latest_detector_path(base_path)
        if p is not None:
            return Detector.load(p, verbose=verbose)
        else:
            return None

    @staticmethod
    def get_latest_detector_path(base_path: Path = STANDARD_BASE_PATH) -> Path | None:
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
        path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")

        return Path(path)

    @staticmethod
    def load_image_dataset(
        base_path: Path,
        target_size: Tuple[int, int],
        n_jobs: Optional[int] = None
    ) -> Tuple[BatchImageArray, Labels, LabelEncoder]:
        """
        Load training images and their labels using parallel processing.
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

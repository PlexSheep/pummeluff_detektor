from __future__ import annotations
import kagglehub
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from datetime import datetime
from tqdm import tqdm
import joblib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union, Callable

# Custom types
ImageArray = np.ndarray  # Shape: (height, width, 3)
BatchImageArray = np.ndarray  # Shape: (n_samples, height, width, 3)
FlatImageArray = np.ndarray  # Shape: (n_samples, height * width * 3)
Labels = np.ndarray  # Shape: (n_samples,)
ProcessArgs = Tuple[str, str, Tuple[int, int]]

SEED = 19


class Metrics:
    class_report: dict[Any, Any] | str
    feature_importance: Any
    seed = SEED

    def __init__(self, class_report, feature_importance) -> None:
        self.class_report = class_report
        self.feature_importance = feature_importance

    def info(self, model: RandomForestClassifier, label_encoder: LabelEncoder) -> str:
        return "no"


def download_dataset() -> Path:
    path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")

    return Path(path)


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


def load_pokemon_dataset(
    base_path: str,
    target_size: Tuple[int, int] = (64, 64),
    n_jobs: Optional[int] = None
) -> Tuple[BatchImageArray, Labels, LabelEncoder]:
    """
    Load Pokemon images and their labels using parallel processing.

    Args:
        base_path: Directory containing Pokemon subdirectories
        target_size: Tuple of (width, height) for resizing images
        n_jobs: Number of parallel processes to use

    Returns:
        Tuple of (image_arrays, labels, label_encoder)
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    # Get all Pokemon classes
    pokemon_classes: List[str] = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    # Prepare arguments for parallel processing
    all_image_args: List[ProcessArgs] = []
    for pokemon in pokemon_classes:
        pokemon_path = os.path.join(base_path, pokemon, pokemon)

        if not os.path.exists(pokemon_path):
            continue

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [
            f for f in os.listdir(pokemon_path)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        for img_file in image_files:
            img_path = os.path.join(pokemon_path, img_file)
            all_image_args.append((img_path, pokemon, target_size))

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


def save_model(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    metrics: Metrics,
    base_path: str = 'models'
) -> Tuple[str, str, str]:
    """
    Save the model, label encoder, and metrics.

    Args:
        model: Trained RandomForestClassifier
        label_encoder: Fitted LabelEncoder
        metrics: Dictionary containing model metrics
        base_path: Directory to save model files

    Returns:
        Tuple of paths (model_path, encoder_path, metrics_path)
    """
    os.makedirs(base_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_path = os.path.join(
        base_path, f'pokemon_classifier_{timestamp}.joblib')
    encoder_path = os.path.join(base_path, f'label_encoder_{timestamp}.joblib')
    metrics_path = os.path.join(base_path, f'metrics_{timestamp}.joblib')

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(metrics, metrics_path)

    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    print(f"Metrics saved to {metrics_path}")

    return model_path, encoder_path, metrics_path


def load_saved_model(
    model_path: str,
    encoder_path: str,
    metrics_path: Optional[str] = None
) -> Tuple[RandomForestClassifier, LabelEncoder, Metrics]:
    """
    Load a saved model and its associated files.

    Args:
        model_path: Path to saved model file
        encoder_path: Path to saved label encoder
        metrics_path: Optional path to saved metrics

    Returns:
        Tuple of (model, label_encoder, metrics)
    """
    model: RandomForestClassifier = joblib.load(model_path)
    label_encoder: LabelEncoder = joblib.load(encoder_path)
    metrics: Metrics = joblib.load(
        metrics_path) if metrics_path else None

    return model, label_encoder, metrics


def get_latest_model_paths(base_path: str = 'models') -> Optional[Tuple[str, str, str]]:
    """
    Find the most recently saved model files.

    Args:
        base_path: Directory containing model files

    Returns:
        Tuple of (model_path, encoder_path, metrics_path) or None if no models found
    """
    if not os.path.exists(base_path):
        return None

    # Find all joblib files
    model_files = []
    encoder_files = []
    metrics_files = []

    for f in os.listdir(base_path):
        if not f.endswith('.joblib'):
            continue

        full_path = os.path.join(base_path, f)
        if f.startswith('pokemon_classifier_'):
            model_files.append(full_path)
        elif f.startswith('label_encoder_'):
            encoder_files.append(full_path)
        elif f.startswith('metrics_'):
            metrics_files.append(full_path)

    # Make sure we have at least one complete set
    if not (model_files and encoder_files and metrics_files):
        return None

    # Get the latest files by modification time
    latest_model = max(model_files, key=os.path.getmtime)
    latest_encoder = max(encoder_files, key=os.path.getmtime)
    latest_metrics = max(metrics_files, key=os.path.getmtime)

    return latest_model, latest_encoder, latest_metrics


def load(force_training=False) -> Tuple[RandomForestClassifier, LabelEncoder, Metrics]:
    """
    Load the most recent model if it exists, otherwise train a new one.
    """
    if not force_training:
        # Try to load the most recent model
        model_paths = get_latest_model_paths()

        if model_paths is not None:
            try:
                print("Found existing model, loading...")
                model, label_encoder, metrics = load_saved_model(*model_paths)
                print("Successfully loaded existing model!")
                return model, label_encoder, metrics
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will train a new model instead.")
        else:
            print("No existing model found. Will train a new one.")

    # If we get here, either no model exists or loading failed
    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Load the dataset using parallel processing
    print("Loading Pokemon dataset...")
    base_path = download_dataset()  # Get the dataset path
    X, y, label_encoder = load_pokemon_dataset(base_path, target_size=(64, 64))

    # Print dataset information
    print("\nDataset Summary:")
    print(f"Number of images: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print("\nClass distribution:")
    for class_name, count in zip(label_encoder.classes_, np.bincount(y)):
        print(f"{class_name}: {count} images")

    # Preprocess the data
    X = X.astype('float32') / 255.0  # Normalize pixel values
    X = X.reshape(X.shape[0], -1)    # Flatten the images

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
    feature_importance = rf_classifier.feature_importances_
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    metrics = Metrics(class_report, feature_importance)

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred,
        classes=label_encoder.classes_,
        title='Pokemon Classification Confusion Matrix'
    )

    # Save the new model
    save_model(rf_classifier, label_encoder, metrics)

    return rf_classifier, label_encoder, metrics

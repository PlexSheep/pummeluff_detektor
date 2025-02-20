import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from pathlib import Path
from skimage.io import collection, imread_collection
import kagglehub
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def download_dataset() -> Path:
    # Download latest version
    path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")
    # path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")

    return Path(path)


def load_dataset(path: Path) -> collection.ImageCollection:
    images: collection.ImageCollection = imread_collection(str(path))
    return images


# def main():
#     dataset_dir = download_dataset()
#     images = load_dataset(dataset_dir)
#     print(images)
#     images[0]


def load_pokemon_dataset(base_path, target_size=(64, 64)):
    """
    Load Pokemon images and their labels from the given directory structure.
    """
    images = []
    labels = []

    pokemon_classes = [d for d in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, d))]

    for pokemon in tqdm(pokemon_classes, desc="Loading Pokemon classes"):
        pokemon_path = os.path.join(base_path, pokemon, pokemon)

        if not os.path.exists(pokemon_path):
            continue

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in os.listdir(pokemon_path)
                       if os.path.splitext(f.lower())[1] in valid_extensions]

        for img_file in tqdm(image_files, desc=f"Loading {pokemon} images", leave=False):
            try:
                img_path = os.path.join(pokemon_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img_array = np.array(img)

                images.append(img_array)
                labels.append(pokemon)

            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue

    X = np.array(images)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return X, y, label_encoder


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """
    Plot confusion matrix using seaborn's heatmap.
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
    plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the dataset
    print("Loading Pokemon dataset...")
    base_path = download_dataset()  # Adjust this to your actual path
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

    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )

    rf_classifier.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = rf_classifier.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    # Plot confusion matrix
    plot_confusion_matrix(
        y_test,
        y_pred,
        classes=label_encoder.classes_,
        title='Pokemon Classification Confusion Matrix'
    )

    # Feature importance visualization
    feature_importance = rf_classifier.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.plot(feature_importance)
    plt.title('Feature Importance in Random Forest Classifier')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()


if __name__ == "__main__":
    main()

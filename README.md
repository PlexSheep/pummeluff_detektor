<div align="center">
    <img alt="icon" src="./docs/logo.png" width="60%"/>
</div>

# 🎵 Pummeluff Detektor ✨

> "Is it a Jigglypuff seen from above?" - Professor Oak, probably

The Pummeluff Detektor is a super sophisticated machine learning system designed to solve one of the most pressing questions in the world: Is that pink blob a Jigglypuff, or is it actually Kirby trying to infiltrate our Pokemon games, or something else entirely?

## 🌟 Features

- 🤖 Advanced ML-powered detection using Random Forest classification
- 🎯 Specifically trained to distinguish Jigglypuff from other round pink creatures
- 📊 Detailed confidence scores and classification reports
- 🎨 Confusion matrix visualization
- 💪 Parallel processing for fast training and image loading
- 📝 Logging system

## 🚀 Quick Start

```bash
# Install the package
pip install pummeluff-detektor

# Run detection on an image
pummeluff-detektor path/to/your/suspicious/pink/blob.jpg

# Train a new model (optional)
pummeluff-detektor --train path/to/your/image.jpg
```

## 🛠️ Command Line Options

```
# just use this, dummy
pummeluff-detektor --help
```

## 📊 How It Works

The Pummeluff Detektor uses a Random Forest Classifier trained on a curated dataset of Pokemon, Kirby, and other images. It's specifically optimized to distinguish Jigglypuff from other things, with special attention paid to telling apart Jigglypuff from its notorious Doppelgänger, Kirby.

### Training Process
1. Images are loaded and preprocessed to a standard size
2. Features are extracted using advanced computer vision techniques
3. A Random Forest model is trained with optimized hyperparameters
4. Performance is evaluated using confusion matrices and classification reports

## 🎭 Emoticon Guide

The detector uses these emoticons to express its confidence:

- ✧٩(•́⌄•́๑)و ✧ - "That's definitely a Jigglypuff!"
- (╥﹏╥) - "No Jigglypuff detected..."
- ¯\\_(Φ ᆺ Φ)_/¯ - "Not quite sure about this one..."
- ヽ(｀⌒´メ)ノ - "IT'S KIRBY!!!"

## 🧪 Development

```bash
# Clone the repository
git clone https://github.com/yourusername/pummeluff-detektor.git
cd pummeluff-detektor

# Install dependencies
poetry install

# Run with verbose output
poetry run pummeluff-detektor -v path/to/image.jpg
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Professor Oak et al. and for his pioneering work in Pokemon identification and ecology
- The Pokemon Company for creating Jigglypuff
- HAL Laboratory for creating Kirby (even though he keeps trying to sneak into other games, especially super smash bros)

## 🤔 FAQ

**Q: Why do we need this?**
A: Because sometimes a pink blob is not just a pink blob.

**Q: Can it detect Jigglypuff from above?**
A: Yes, but please make sure it's not just a regular pokeball.

**Q: What if my Jigglypuff is using Transform?**
A: Then you probably have a Ditto. This detector can't help you with that.

**Q: Will this make me sleepy?**
A: This detector does not have sound, so it cannot play the famous [song of Jigglypuff](https://bulbapedia.bulbagarden.net/wiki/EP045).

## 🖼️ Dataset with the images used for training

The dataset is custom made and can be found
[here](https://www.kaggle.com/datasets/plexsheep/jigglypuff-detection-data)
(I hope this link works, otherwise you can contact me personally).

The pictures are combined together from these datasets:

- [CS63 Pokemon Dataset](https://www.kaggle.com/datasets/alinapalacios/cs63-pokemon-dataset)
- [kirby dataset](https://www.kaggle.com/datasets/turbihao/nintendo-kirby-dataset)

## 🐛 Known Issues

- Performance may degrade if the subject is using Minimize
- Not guaranteed to work on Jigglypuff wearing sunglasses, or on very musuclar Jigglypuff

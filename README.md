# Image Sorter

An automated image categorization tool that uses deep learning to sort images into predefined categories.

## Overview

This application uses a ResNet50-based neural network to classify and sort images into categories. It works in two phases:
1. **Training Phase**: The model learns from categorized images in the `dataset` directory
2. **Sorting Phase**: The trained model automatically sorts images from `myImages` into appropriate categories in the `finalSorted` directory

## Features

- Uses transfer learning with pre-trained ResNet50 model
- Automatically creates category folders in the destination directory
- Preserves original images (copies rather than moves)
- Supports .jpg, .jpeg, and .png file formats
- Configurable via environment variables or command line arguments
- Saves the trained model for future use without retraining

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`:
  ```
  tensorflow>=2.4.0
  scikit-learn>=0.24.0
  pillow>=8.0.0
  numpy>=1.19.0
  python-dotenv>=0.19.0
  ```

## Directory Structure

```
├── dataset/           # Training data (organized by category)
│   ├── category1/     # Each subfolder represents a category
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   ├── category2/
│   │   ├── img3.jpg
│   │   └── ...
├── myImages/          # Images to be sorted
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── finalSorted/       # Destination for sorted images (created automatically)
│   ├── category1/
│   ├── category2/
│   └── ...
├── model/             # Saved model and metadata (created automatically)
│   ├── image_sorter.h5
│   ├── label_encoder.pkl
│   └── categories.pkl
├── .env               # Environment variables (optional)
├── main.py            # Original script
└── image_sorting_reuse.py  # Enhanced version with model saving/loading
```

## Usage

### Basic Usage

1. **Prepare your dataset**:
   - Create a `dataset` directory
   - Inside it, create subdirectories for each category (e.g., `dogs`, `cats`, `landscapes`)
   - Place sample images in each category directory

2. **Prepare images to sort**:
   - Place images you want to sort in the `myImages` directory

3. **Run the script**:
   ```
   python image_sorting_reuse.py
   ```

4. **Find sorted images**:
   - Check the `finalSorted` directory for your categorized images

### Command Line Arguments

The enhanced version (`image_sorting_reuse.py`) supports these arguments:

- `--train`: Force training a new model even if one already exists
- `--input [path]`: Specify a custom directory containing images to sort (default: `./myImages`)

Example:
```
python image_sorting_reuse.py --train --input ./vacation_photos
```

### Configuration

You can customize the application by setting these environment variables in a `.env` file:

```
DATASET_DIR=./dataset
FINAL_SORTED_DIR=./finalSorted
MY_IMAGES_DIR=./myImages
MODEL_PATH=./model/image_sorter.h5
LABEL_ENCODER_PATH=./model/label_encoder.pkl
CATEGORIES_PATH=./model/categories.pkl
BATCH_SIZE=32
EPOCHS=10
VALIDATION_SPLIT=0.2
LEARNING_RATE=0.001
```

## How It Works

1. **Feature Extraction**: Uses ResNet50 (pre-trained on ImageNet) to extract high-level features from images
2. **Model Training**: Trains a classifier on top of these features to recognize your specific categories
3. **Prediction**: Applies the trained model to new images and sorts them based on predictions
4. **Persistence**: Saves the trained model and metadata for future use

## Performance Considerations

- **Dataset Quality**: Include diverse, high-quality examples for each category
- **Dataset Size**: Aim for at least 20-30 images per category for decent results
- **Training Time**: Depends on dataset size and hardware
- **Accuracy**: Heavily dependent on how distinct your categories are and dataset quality

## Troubleshooting

- **Incorrect Classifications**: Add more training examples to the category that's being misclassified
- **Model Loading Errors**: Delete the `model` directory and retrain by using the `--train` flag
- **Memory Errors**: Reduce `BATCH_SIZE` in the `.env` file

## Contributing

Feel free to submit issues or pull requests to improve this project.

## License

This project is open source and available under the [MIT License](LICENSE).
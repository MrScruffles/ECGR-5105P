import os
import shutil
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path configuration from environment variables
DATASET_DIR = os.getenv('DATASET_DIR', './dataset')
FINAL_SORTED_DIR = os.getenv('FINAL_SORTED_DIR', './finalSorted')
MY_IMAGES_DIR = os.getenv('MY_IMAGES_DIR', './myImages')
MODEL_PATH = os.getenv('MODEL_PATH', './model/image_sorter.h5')
LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', './model/label_encoder.pkl')
CATEGORIES_PATH = os.getenv('CATEGORIES_PATH', './model/categories.pkl')

# Training parameters from environment variables
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 10))
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', 0.2))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))

def extract_features(image_path, target_size=(224, 224)):
    """Extract features from an image using a pre-trained model."""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_dataset(directory):
    """Load images and labels from directory structure."""
    images = []
    labels = []
    categories = []
    
    # Identify categories based on subdirectories
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            categories.append(category)
            
            # Get all images in this category
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    features = extract_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(category)
    
    return np.vstack(images), np.array(labels), categories

def build_model(num_classes):
    """Build and compile the model."""
    # Use ResNet50 as base model
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_save_model():
    """Train the image sorting model and save it."""
    print("Loading and preprocessing dataset images...")
    X, y, categories = get_dataset(DATASET_DIR)
    
    # Convert categorical labels to numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # Build and train the model
    print(f"Building model for {len(categories)} categories: {categories}")
    model = build_model(len(categories))
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save the model and associated data
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    # Save the label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Make sure categories is a list to avoid any type issues later
    categories_list = list(categories)
    
    # Save the categories
    with open(CATEGORIES_PATH, 'wb') as f:
        pickle.dump(categories_list, f)
    
    print("Model and associated data saved successfully.")
    
    return model, label_encoder, categories_list

def load_saved_model():
    """Load the previously trained model and associated data."""
    print(f"Loading saved model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    # Load the label encoder
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load the categories
    with open(CATEGORIES_PATH, 'rb') as f:
        categories = pickle.load(f)
    
    # Ensure categories is a list
    if not isinstance(categories, list):
        categories = list(categories)
    
    print(f"Model loaded successfully for {len(categories)} categories: {categories}")
    
    return model, label_encoder, categories

def sort_images(model, label_encoder, categories, input_dir=None):
    """Sort images into finalSorted directory based on model predictions."""
    # If no input directory specified, use the default
    if input_dir is None:
        input_dir = MY_IMAGES_DIR
    
    print(f"Sorting images from {input_dir}...")
    
    # Create category subdirectories in finalSorted if they don't exist
    for category in categories:
        os.makedirs(os.path.join(FINAL_SORTED_DIR, category), exist_ok=True)
    
    # Convert categories to a list if it's not already
    if not isinstance(categories, list):
        categories = list(categories)
    
    # Process each image in the input directory
    sorted_count = 0
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Extract features
                features = extract_features(img_path)
                if features is None:
                    continue
                
                # Predict category
                prediction = model.predict(features)
                category_idx = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([category_idx])[0]
                
                # The predicted_label should directly correspond to one of the categories
                if predicted_label in categories:
                    category = predicted_label
                else:
                    # If not, try to find it by index
                    category = categories[category_idx]
                
                # Copy the image to the appropriate category in finalSorted
                dest_path = os.path.join(FINAL_SORTED_DIR, category, img_name)
                shutil.copy(img_path, dest_path)
                sorted_count += 1
                print(f"Sorted {img_name} into {category}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    print(f"Successfully sorted {sorted_count} images.")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Sort images into categories using a trained model.')
    parser.add_argument('--train', action='store_true', help='Force training a new model')
    parser.add_argument('--input', type=str, help='Directory containing images to sort (default: ./myImages)')
    
    args = parser.parse_args()
    
    # Check if we need to train a new model or load an existing one
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(CATEGORIES_PATH)
    
    if args.train or not model_exists:
        # Train and save a new model
        model, label_encoder, categories = train_and_save_model()
    else:
        # Load existing model
        try:
            model, label_encoder, categories = load_saved_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead.")
            model, label_encoder, categories = train_and_save_model()
    
    # Sort images
    input_dir = args.input if args.input else MY_IMAGES_DIR
    sort_images(model, label_encoder, categories, input_dir)
    
    print("Image sorting completed!")

if __name__ == "__main__":
    main()
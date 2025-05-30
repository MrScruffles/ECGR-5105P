# there is 3 folders
# ./dataset, ./finalSorted, ./myImages

# now take all the images in side of the dataset and use it to train a model to help me make it so my ./myImages images when ran also get sorted itno the finalSorted after its trained.
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Path configuration
DATASET_DIR = './dataset'
FINAL_SORTED_DIR = './finalSorted'
MY_IMAGES_DIR = './myImages'

# Create finalSorted directory if it doesn't exist
os.makedirs(FINAL_SORTED_DIR, exist_ok=True)

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
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the image sorting model."""
    print("Loading and preprocessing dataset images...")
    X, y, categories = get_dataset(DATASET_DIR)
    
    # Convert categorical labels to numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Build and train the model
    print(f"Building model for {len(categories)} categories: {categories}")
    model = build_model(len(categories))
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val)
    )
    
    return model, label_encoder, categories

def sort_images(model, label_encoder, categories):
    """Sort images from myImages into finalSorted directory."""
    print("Sorting your personal images...")
    
    # Create category subdirectories in finalSorted if they don't exist
    for category in categories:
        os.makedirs(os.path.join(FINAL_SORTED_DIR, category), exist_ok=True)
    
    # Process each image in myImages
    sorted_count = 0
    for img_name in os.listdir(MY_IMAGES_DIR):
        img_path = os.path.join(MY_IMAGES_DIR, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract features
            features = extract_features(img_path)
            if features is None:
                continue
            
            # Predict category
            prediction = model.predict(features)
            category_idx = np.argmax(prediction)
            category = categories[label_encoder.inverse_transform([category_idx])[0]]
            
            # Copy the image to the appropriate category in finalSorted
            dest_path = os.path.join(FINAL_SORTED_DIR, category, img_name)
            shutil.copy(img_path, dest_path)
            sorted_count += 1
            print(f"Sorted {img_name} into {category}")
    
    print(f"Successfully sorted {sorted_count} images.")

if __name__ == "__main__":
    print("Starting image sorting process...")
    
    # Check if dataset directory has subdirectories
    has_categories = False
    for item in os.listdir(DATASET_DIR):
        if os.path.isdir(os.path.join(DATASET_DIR, item)):
            has_categories = True
            break
    
    if not has_categories:
        print("Error: The dataset directory must contain subdirectories representing categories.")
        exit(1)
    
    # Train the model
    model, label_encoder, categories = train_model()
    
    # Sort images
    sort_images(model, label_encoder, categories)
    
    print("Image sorting completed!")
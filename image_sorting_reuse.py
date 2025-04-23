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

load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR', './dataset')
FINAL_SORTED_DIR = os.getenv('FINAL_SORTED_DIR', './finalSorted')
MY_IMAGES_DIR = os.getenv('MY_IMAGES_DIR', './myImages')
MODEL_PATH = os.getenv('MODEL_PATH', './model/image_sorter.h5')
LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', './model/label_encoder.pkl')
CATEGORIES_PATH = os.getenv('CATEGORIES_PATH', './model/categories.pkl')

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 10))
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', 0.2))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))

def extract_features(image_path, target_size=(224, 224)):
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
    images = []
    labels = []
    categories = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            categories.append(category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    features = extract_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(category)
    return np.vstack(images), np.array(labels), categories

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    print("Loading and preprocessing dataset images...")
    X, y, categories = get_dataset(DATASET_DIR)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Building model for {len(categories)} categories: {categories}")
    model = build_model(len(categories))
    print("Training model...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    categories_list = list(categories)
    with open(CATEGORIES_PATH, 'wb') as f:
        pickle.dump(categories_list, f)
    print("Model and associated data saved successfully.")
    return model, label_encoder, categories_list

def load_saved_model():
    print(f"Loading saved model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(CATEGORIES_PATH, 'rb') as f:
        categories = pickle.load(f)
    if not isinstance(categories, list):
        categories = list(categories)
    print(f"Model loaded successfully for {len(categories)} categories: {categories}")
    return model, label_encoder, categories

def sort_images(model, label_encoder, categories, input_dir=None):
    if input_dir is None:
        input_dir = MY_IMAGES_DIR
    print(f"Sorting images from {input_dir}...")
    for category in categories:
        os.makedirs(os.path.join(FINAL_SORTED_DIR, category), exist_ok=True)
    if not isinstance(categories, list):
        categories = list(categories)
    sorted_count = 0
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                features = extract_features(img_path)
                if features is None:
                    continue
                prediction = model.predict(features)
                category_idx = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([category_idx])[0]
                if predicted_label in categories:
                    category = predicted_label
                else:
                    category = categories[category_idx]
                dest_path = os.path.join(FINAL_SORTED_DIR, category, img_name)
                shutil.copy(img_path, dest_path)
                sorted_count += 1
                print(f"Sorted {img_name} into {category}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    print(f"Successfully sorted {sorted_count} images.")

def main():
    parser = argparse.ArgumentParser(description='Sort images into categories using a trained model.')
    parser.add_argument('--train', action='store_true', help='Force training a new model')
    parser.add_argument('--input', type=str, help='Directory containing images to sort (default: ./myImages)')
    args = parser.parse_args()
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(CATEGORIES_PATH)
    if args.train or not model_exists:
        model, label_encoder, categories = train_and_save_model()
    else:
        try:
            model, label_encoder, categories = load_saved_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead.")
            model, label_encoder, categories = train_and_save_model()
    input_dir = args.input if args.input else MY_IMAGES_DIR
    sort_images(model, label_encoder, categories, input_dir)
    print("Image sorting completed!")

if __name__ == "__main__":
    main()
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

load_dotenv()

DATASET_DIR = os.getenv('DATASET_DIR', './dataset')
FINAL_SORTED_DIR = os.getenv('FINAL_SORTED_DIR', './finalSorted')
MY_IMAGES_DIR = os.getenv('MY_IMAGES_DIR', './myImages')
MODEL_PATH = os.getenv('MODEL_PATH', './model/image_sorter.h5')
LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', './model/label_encoder.pkl')
CATEGORIES_PATH = os.getenv('CATEGORIES_PATH', './model/categories.pkl')

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 10))
VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', 0.2))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))

def extract_features(image_path, target_size=(224, 224)):
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
    images = []
    labels = []
    categories = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            categories.append(category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    features = extract_features(img_path)
                    if features is not None:
                        images.append(features)
                        labels.append(category)
    return np.vstack(images), np.array(labels), categories

def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model():
    print("Loading and preprocessing dataset images...")
    X, y, categories = get_dataset(DATASET_DIR)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Building model for {len(categories)} categories: {categories}")
    model = build_model(len(categories))
    print("Training model...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    categories_list = list(categories)
    with open(CATEGORIES_PATH, 'wb') as f:
        pickle.dump(categories_list, f)
    print("Model and associated data saved successfully.")
    return model, label_encoder, categories_list

def load_saved_model():
    print(f"Loading saved model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(CATEGORIES_PATH, 'rb') as f:
        categories = pickle.load(f)
    if not isinstance(categories, list):
        categories = list(categories)
    print(f"Model loaded successfully for {len(categories)} categories: {categories}")
    return model, label_encoder, categories

def sort_images(model, label_encoder, categories, input_dir=None):
    if input_dir is None:
        input_dir = MY_IMAGES_DIR
    print(f"Sorting images from {input_dir}...")
    for category in categories:
        os.makedirs(os.path.join(FINAL_SORTED_DIR, category), exist_ok=True)
    if not isinstance(categories, list):
        categories = list(categories)
    sorted_count = 0
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                features = extract_features(img_path)
                if features is None:
                    continue
                prediction = model.predict(features)
                category_idx = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([category_idx])[0]
                if predicted_label in categories:
                    category = predicted_label
                else:
                    category = categories[category_idx]
                dest_path = os.path.join(FINAL_SORTED_DIR, category, img_name)
                shutil.copy(img_path, dest_path)
                sorted_count += 1
                print(f"Sorted {img_name} into {category}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    print(f"Successfully sorted {sorted_count} images.")

def main():
    parser = argparse.ArgumentParser(description='Sort images into categories using a trained model.')
    parser.add_argument('--train', action='store_true', help='Force training a new model')
    parser.add_argument('--input', type=str, help='Directory containing images to sort (default: ./myImages)')
    args = parser.parse_args()
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(CATEGORIES_PATH)
    if args.train or not model_exists:
        model, label_encoder, categories = train_and_save_model()
    else:
        try:
            model, label_encoder, categories = load_saved_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model instead.")
            model, label_encoder, categories = train_and_save_model()
    input_dir = args.input if args.input else MY_IMAGES_DIR
    sort_images(model, label_encoder, categories, input_dir)
    print("Image sorting completed!")

if __name__ == "__main__":
    main()
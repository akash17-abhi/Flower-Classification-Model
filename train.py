import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
import json
from tqdm import tqdm

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 104  # Kaggle flower dataset has 104 classes

def download_kaggle_dataset():
    """Download flower dataset from Kaggle"""
    try:
        print("Downloading Kaggle flower dataset...")
        os.system('kaggle datasets download -d iamaditey/flower-classification-dataset')
        # Extract and organize the dataset
        import zipfile
        with zipfile.ZipFile('flower-classification-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset')
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_data_from_directory(data_dir):
    """Load data from organized directory structure"""
    classes = []
    images = []
    labels = []
    
    for i, class_name in enumerate(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            print(f"Loading {class_name}...")
            for img_file in tqdm(os.listdir(class_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = tf.keras.preprocessing.image.load_img(
                            os.path.join(class_path, img_file),
                            target_size=(IMG_HEIGHT, IMG_WIDTH)
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        images.append(img_array)
                        labels.append(i)
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
    
    return np.array(images), np.array(labels), classes

def create_model(num_classes):
    """Create MobileNetV2 transfer learning model"""
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
                             include_top=False, 
                             weights='imagenet')
    base_model.trainable = False  # Freeze base model initially
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def main():
    # Try to use Kaggle dataset, fallback to synthetic if fails
    use_real_data = False
    
    if os.path.exists('dataset/flowers'):
        print("Loading existing dataset...")
        X, y, classes = load_data_from_directory('dataset/flowers')
        if len(X) > 0:
            use_real_data = True
            print(f"Loaded {len(X)} images of {len(classes)} classes")
    else:
        # Try downloading
        if download_kaggle_dataset():
            X, y, classes = load_data_from_directory('dataset/flowers')
            if len(X) > 0:
                use_real_data = True
    
    if not use_real_data:
        print("\nUsing synthetic data for demonstration (download Kaggle dataset for real training)")
        # Fallback to synthetic data
        classes = ['daisy', 'rose', 'sunflower', 'tulip', 'dandelion']
        num_classes = len(classes)
        
        def generate_synthetic_data(num_samples_per_class=500):
            X = []
            y = []
            for i, cls in enumerate(classes):
                for _ in range(num_samples_per_class):
                    img = np.random.randint(100, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                    if cls == 'daisy': 
                        img[:,:,1] = np.minimum(255, img[:,:,1] + 60)
                    elif cls == 'rose': 
                        img[:,:,0] = np.minimum(255, img[:,:,0] + 70)
                    elif cls == 'sunflower': 
                        img[:,:,0:2] = np.minimum(255, img[:,:,0:2] + 50)
                    elif cls == 'tulip': 
                        img[:,:,2] = np.minimum(255, img[:,:,2] + 60)
                    elif cls == 'dandelion': 
                        img[:,:,1] = np.minimum(255, img[:,:,1] + 80)
                    X.append(img)
                    y.append(i)
            return np.array(X), np.array(y)
        
        print("Generating synthetic training data...")
        X, y = generate_synthetic_data(500)
        num_classes = len(classes)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # One-hot encode
    y_train = to_categorical(y_train, num_classes if use_real_data else len(classes))
    y_test = to_categorical(y_test, num_classes if use_real_data else len(classes))
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Data augmentation (only for real data)
    if use_real_data:
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Create model
    model, base_model = create_model(num_classes if use_real_data else len(classes))
    
    # Compile model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    print("Training model...")
    if use_real_data:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr]
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test)
        )
    
    # Fine-tune top layers
    print("Fine-tuning model...")
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fine-tune training
    if use_real_data:
        model.fit(
            train_generator,
            epochs=10,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr]
        )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('models/flower_model.h5')
    print("Model saved to models/flower_model.h5")
    
    # Save class labels
    class_labels = {'classes': classes, 'num_classes': len(classes)}
    with open('models/class_labels.json', 'w') as f:
        json.dump(class_labels, f)
    print("Class labels saved to models/class_labels.json")
    print(f"Classes ({len(classes)} total): {classes[:10]}{'...' if len(classes) > 10 else ''}")

if __name__ == '__main__':
    main()


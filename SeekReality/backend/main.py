# Import necessary modules
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import numpy as np
import cv2
import os
import tempfile
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import LeakyReLU, LSTM, Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define image dimensions
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

### Define the Classifier class
class Classifier:
    """
    A class for defining a generic classifier.
    """
    def __init__(self):
        """
        Constructor method to initialize the classifier.
        """
        self.model = None
    
    def predict(self, x):
        """
        Method to make predictions using the classifier.

        Args:
            x: Input data for prediction.

        Returns:
            Prediction made by the classifier.
        """
        return self.model.predict(x)
    
    def fit(self, x, y):
        """
        Method to fit the classifier to the training data.

        Args:
            x: Input data for training.
            y: Target labels for training.

        Returns:
            Training loss and metrics.
        """
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        """
        Method to evaluate the accuracy of the classifier.

        Args:
            x: Input data for evaluation.
            y: Target labels for evaluation.

        Returns:
            Accuracy of the classifier.
        """
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        """
        Method to load pre-trained weights into the classifier.

        Args:
            path: Path to the pre-trained weights.
        """
        self.model.load_weights(path)


### Define the Meso4 class
class Meso4(Classifier):
    """
    A class for defining the Meso4 model for deepfake detection.
    """
    def __init__(self, learning_rate=0.001):
        """
        Constructor method to initialize the Meso4 model.

        Args:
            learning_rate: Learning rate for model optimization.
        """
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])
    
    def init_model(self): 
        """
        Method to initialize the Meso4 model architecture.

        Returns:
            Initialized Meso4 model.
        """
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


### Define the MesoLSTM class for video analysis
class MesoLSTM(Classifier):
    """
    A class for defining the MesoLSTM model for deepfake video detection.
    This model combines the Meso4 CNN with an LSTM for temporal analysis.
    """
    def __init__(self, learning_rate=0.001, sequence_length=20):
        """
        Constructor method to initialize the MesoLSTM model.

        Args:
            learning_rate: Learning rate for model optimization.
            sequence_length: Number of frames to analyze in sequence.
        """
        self.sequence_length = sequence_length
        self.meso4 = Meso4()
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    
    def init_model(self):
        """
        Method to initialize the MesoLSTM model architecture.

        Returns:
            Initialized MesoLSTM model.
        """
        # Get the Meso4 feature extractor (remove the last layer)
        feature_extractor = Model(inputs=self.meso4.model.inputs, 
                                  outputs=self.meso4.model.layers[-3].output)
        
        # Input shape for sequence of frames
        input_shape = (self.sequence_length, 
                       image_dimensions['height'],
                       image_dimensions['width'],
                       image_dimensions['channels'])
        
        # Define the LSTM model
        inputs = Input(shape=input_shape)
        
        # Extract features from each frame using the Meso4 feature extractor
        features = TimeDistributed(feature_extractor)(inputs)
        
        # Process the sequence of features with LSTM
        lstm_out = LSTM(32, return_sequences=False)(features)
        
        # Final classification layer
        outputs = Dense(1, activation='sigmoid')(lstm_out)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def load_meso4_weights(self, path):
        """
        Method to load pre-trained weights for the Meso4 feature extractor.

        Args:
            path: Path to the pre-trained weights.
        """
        self.meso4.load(path)
        # Update the feature extractor in the LSTM model
        feature_extractor = Model(inputs=self.meso4.model.inputs, 
                                  outputs=self.meso4.model.layers[-3].output)
        self.model.layers[1].layer = feature_extractor



# Load the pre-trained MesoNet model
meso = Meso4()
meso.load('./weights/Meso4_DF')


# Initialize the MesoLSTM model (for video analysis)
meso_lstm = MesoLSTM()
meso_lstm.load_meso4_weights('./weights/Meso4_DF')


def process_image(img_bytes):
    """
    Process an image for deepfake detection.

    Args:
        img_bytes: Image bytes data.

    Returns:
        Dictionary containing prediction result, confidence score, and the processed image.
    """
    # Convert the bytes to an image
    img = load_img(io.BytesIO(img_bytes), target_size=(256, 256))
    
    # Process the image
    img_array = img_to_array(img)
    
    # Preprocess the image
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = meso.predict(img_array)[0][0]
    
    # Determine the result
    if prediction > 0.5:
        result = "Real"
        confidence = prediction
    else:
        result = "Fake"
        confidence = 1 - prediction  # Invert for "fakeness" confidence
    
    # Encode the image to base64 format
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return {
        'result': result,
        'confidence': float(confidence),
        'image': img_base64
    }


def extract_frames(video_bytes, num_frames=20):
    """
    Extract frames from a video file.

    Args:
        video_bytes: Video bytes data.
        num_frames: Number of frames to extract.

    Returns:
        List of extracted frames as numpy arrays.
    """
    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name
    
    # Open the video file
    cap = cv2.VideoCapture(temp_file_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        os.unlink(temp_file_path)
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print("Error: Video has no frames")
        os.unlink(temp_file_path)
        return []
    
    # Calculate frame indices to extract (evenly distributed)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        # Get frames more concentrated in the beginning and middle
        # This helps catch deepfakes that might be more prominent in certain segments
        indices_part1 = [int(i * total_frames / (2 * num_frames)) for i in range(num_frames // 2)]
        indices_part2 = [int(total_frames // 2 + i * total_frames / (2 * num_frames)) for i in range(num_frames // 2)]
        frame_indices = indices_part1 + indices_part2
    
    frames = []
    face_frames = []  # Specifically store frames with faces
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Check if the frame has faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # If this frame has faces, prioritize it
            if len(faces) > 0:
                face_frames.append((idx, frame))
            
            # Resize the frame
            frame = cv2.resize(frame, (image_dimensions['width'], image_dimensions['height']))
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize the frame
            frame = frame / 255.0
            frames.append(frame)
    
    # If we found face frames but not enough regular frames, use more face frames
    if len(frames) < num_frames // 2 and face_frames:
        face_indices = [idx for idx, _ in face_frames]
        for idx, frame in face_frames:
            if idx not in frame_indices:
                # Resize and add this face frame
                frame = cv2.resize(frame, (image_dimensions['width'], image_dimensions['height']))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frames.append(frame)
                if len(frames) >= num_frames:
                    break
    
    # Release the video capture object and delete the temporary file
    cap.release()
    os.unlink(temp_file_path)
    
    return frames


def process_video(video_bytes):
    """
    Process a video for deepfake detection.

    Args:
        video_bytes: Video bytes data.

    Returns:
        Dictionary containing prediction results for the video.
    """
    # Extract frames from the video
    frames = extract_frames(video_bytes, num_frames=30)  # Increased number of frames
    
    if not frames:
        return {
            'error': 'Could not extract frames from the video',
            'result': 'Unknown',
            'confidence': 0.0
        }
    
    # If we don't have enough frames, pad the list
    if len(frames) < meso_lstm.sequence_length:
        # Duplicate the last frame to reach the required sequence length
        frames.extend([frames[-1]] * (meso_lstm.sequence_length - len(frames)))
    
    # Analyze using frame-by-frame approach with enhanced detection
    frame_predictions = []
    frame_images = []
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for i, frame in enumerate(frames):
        # Convert to uint8 for OpenCV operations
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Detect faces
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_preds = []
        annotated_frame = frame_uint8.copy()
        
        # Process each face if found
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extract and process face
                face = frame_uint8[y:y+h, x:x+w]
                face = cv2.resize(face, (256, 256))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)
                
                face_pred = float(meso.predict(face)[0][0])  # Convert to Python float
                face_preds.append(face_pred)
                
                # Draw rectangle
                color_bgr = (0, 255, 0) if face_pred > 0.6 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color_bgr, 2)
                
                # Add text with confidence
                conf_text = f"R:{face_pred:.2f}" if face_pred > 0.6 else f"F:{1-face_pred:.2f}"
                cv2.putText(annotated_frame, conf_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # Always perform full frame analysis too
        frame_array = np.expand_dims(frame, axis=0)
        full_pred = float(meso.predict(frame_array)[0][0])  # Convert to Python float
        
        # Combine predictions, prioritize face predictions
        if face_preds:
            # Use the minimum face prediction (more conservative approach)
            frame_pred = min(face_preds)
        else:
            frame_pred = full_pred
        
        frame_predictions.append(frame_pred)
        
        # Store annotated frame for display
        if len(faces) > 0:
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', annotated_frame_rgb)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            frame_images.append({
                'image': img_base64,
                'prediction': float(frame_pred),  # Convert to Python float
                'result': 'Real' if frame_pred > 0.6 else 'Fake',
                'color': 'green' if frame_pred > 0.6 else 'red',
                'has_faces': len(faces) > 0
            })
    
    # Analyze prediction distribution for more robust results
    frame_predictions = np.array(frame_predictions)
    
    # More sophisticated decision logic
    avg_prediction = float(np.mean(frame_predictions))  # Convert to Python float
    median_prediction = float(np.median(frame_predictions))  # Convert to Python float
    lower_quartile = float(np.percentile(frame_predictions, 25))  # Convert to Python float
    
    # Consider temporal consistency for more robust detection
    prediction_diffs = np.abs(np.diff(frame_predictions))
    temporal_consistency = float(np.mean(prediction_diffs))  # Convert to Python float
    
    # If many predictions fluctuate wildly, it's more likely to be fake
    consistency_factor = max(0, 1 - (temporal_consistency * 5))
    
    # Combined decision metric with higher weight on lower quartile
    # Lower quartile being low is a stronger indicator of fakeness
    decision_metric = (0.3 * avg_prediction + 0.3 * median_prediction + 
                      0.4 * lower_quartile) * consistency_factor
    
    # Select key frames for display (use frames with most extreme predictions)
    sorted_indices = np.argsort(frame_predictions)
    key_frame_indices = [
        sorted_indices[0],  # Most fake frame
        sorted_indices[len(sorted_indices)//4],
        sorted_indices[len(sorted_indices)//2],  # Median frame
        sorted_indices[3*len(sorted_indices)//4],
        sorted_indices[-1]  # Most real frame
    ]
    
    # Sort indices for chronological display
    key_frame_indices.sort()
    
    # Select the key frames for display
    key_frames = [frame_images[i] for i in key_frame_indices if i < len(frame_images)]
    
    # Make final determination with adjusted threshold
    if decision_metric > 0.6:  # Higher threshold for claiming "real"
        result = "Real"
        color = "green"
        confidence = float(decision_metric)  # Convert to Python float
    else:
        result = "Fake"
        color = "red"
        confidence = float(1 - decision_metric)  # Convert to Python float
    
    return {
        'result': result,
        'confidence': confidence,
        'color': color,
        'frame_results': key_frames,
        'frame_predictions': [float(x) for x in frame_predictions.tolist()],  # Convert to Python floats
        'consistency': float(consistency_factor),  # Convert to Python float
        'statistics': {
            'average': avg_prediction,
            'median': median_prediction,
            'lower_quartile': lower_quartile,
            'temporal_consistency': temporal_consistency
        }
    }


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the image file
        img_bytes = file.read()
        
        # Process the image
        result = process_image(img_bytes)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the video file
        video_bytes = file.read()
        
        # Process the video
        result = process_video(video_bytes)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, port=5000)

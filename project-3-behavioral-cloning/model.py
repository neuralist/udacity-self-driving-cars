# Imports
import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

### Command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 512, "The batch size.")
flags.DEFINE_float('learning_rate', 0.0002, "The lerarning rate for the Adam optimizer.")


### ------------------------------------------------
### Step 1: Load Dataset
### ------------------------------------------------
# Data collected from both tracks will be used.
def read_track_log_data(track_name):
    
    log_dir = "./{}-log/".format(track_name)
    log_file = "{}-log.csv".format(track_name)
    
    # Read track log csv
    driving_records = pd.read_csv(log_dir + log_file)
    
    # Expand image paths to full paths
    driving_records['center'] = log_dir + driving_records['center'].str.strip()
    driving_records['left'] = log_dir + driving_records['left'].str.strip()
    driving_records['right'] = log_dir + driving_records['right'].str.strip()

    return driving_records

# Read (Udacity-provided) driving log from Lake Track
lake_track_driving_records = read_track_log_data("lake-track")

# Read driving log from Jungle Track
jungle_track_driving_records = read_track_log_data("jungle-track")

# Merge track data
driving_records = lake_track_driving_records.append(jungle_track_driving_records)


### ------------------------------------------------
### Step 2: Prepare Dataset
### ------------------------------------------------

#----- 2.1 Deal with angle bias

# Drop some zero angle examples
zero_angle = driving_records[np.abs(driving_records['steering']) == 0.0]

# Drop some examples
drop_factor = 0.5
drop_indices = np.random.choice(zero_angle.index, int(np.ceil(drop_factor*len(zero_angle))), replace=False)
driving_records = driving_records.drop(drop_indices)


#----- 2.2 Create recovery data

# Use side camera images
# Assume correction is a constant
steering_correction = 0.25

### Get center camera images and corresponding steering angles
driving_log = driving_records[['center', 'steering']].copy()
driving_log.columns = ['camera_image', 'steering_angle']

# Extract records defining turns
turns_left_cam = driving_records[['left', 'steering']].copy()
turns_left_cam.columns = ['camera_image', 'steering_angle']
turns_right_cam = driving_records[['right', 'steering']].copy()
turns_right_cam.columns = ['camera_image', 'steering_angle']

# Correct steering angles
turns_left_cam.loc[:, 'steering_angle'] += steering_correction
turns_right_cam.loc[:, 'steering_angle'] -= steering_correction

# Append recovery data to driving records
driving_log = pd.concat([driving_log, turns_left_cam, turns_right_cam])


#----- 2.3 Image perturbations

# Flip image
def apply_random_flip(img, angle):
    
    # Flip image randomly to simulate driving in the other direction
    flip = np.random.randint(0,2)
    
    if flip:
        img = cv2.flip(img, 1)
        
        # Steering angle must be inverted if image was flipped
        angle = -angle
        
    return img, angle

# Change image brightness randomly
def apply_random_brightness(img):
    
    # Convert to HSV space, where brightness is controlled in the V-channel
    hsv_img = img
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)

    # Adjust V-value randomly
    hsv_img[:,:,2] *= np.random.uniform(0.3, 1.3)
    hsv_img = np.clip(hsv_img, 0, 255)

    # Return image to RGB
    return cv2.cvtColor(hsv_img.astype('uint8'), cv2.COLOR_HSV2RGB)

# Translate image in random direction (x,y)
def apply_random_translation(img, angle, max_translation=20):
    
    rows, cols, _ = img.shape
    
    # Translate in random direction (t_x, t_y)
    t_x = max_translation*np.random.uniform(-1.0, 1.0)
    t_y = max_translation*np.random.uniform(-1.0, 1.0)

    # The affine operator for translation is [1 0 t_x]
    #                                        [0 1 t_y]
    #
    affine_operator = np.float32([[1, 0, t_x],[0, 1, t_y]])
    
    # Adjust steering angle after horizontal translation
    # 15 pixles corresponds roughly to the shift from left to right camera
    angle += (2*steering_correction/15.0)*t_x
    
    return cv2.warpAffine(img, affine_operator, (cols, rows)), angle


### ------------------------------------------------
### Step 3: Create Data Model
### ------------------------------------------------

# Hyperparameters
DROPOUT = 0.5
L2_REGULARIZATION = 0.001

adam_optimizer = optimizers.Adam(lr=FLAGS.learning_rate)


#----- 3.1 Define CNN

# This definition has to be placed here for imports to work!
def resize_image(img):
    from keras.backend import tf as ktf
    
    # Cropped image size is (80, 320)
    # Shrink image by a factor 2
    return ktf.image.resize_images(img, (40, 160))


### Define CNN model
model = Sequential()

### Pre-processing layers
# Crop
model.add(Cropping2D(cropping=((55, 25), (0,0)), trainable=False, input_shape=(160, 320, 3)))

# Resize
model.add(Lambda(resize_image, trainable=False))

# Normalize [-0.5, 0.5]
model.add(Lambda(lambda x: (x/255.0) - 0.5, trainable=False))


### Build Neural Network 
# Convolutional layers
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
          
# FC layers
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))

model.add(Dense(50, W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))

model.add(Dense(10, W_regularizer=l2(L2_REGULARIZATION)))
model.add(Dropout(DROPOUT))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


#----- 3.2 Train model

### Define batch generator
def get_batch(driving_log, batch_size=128):

    while 1: # Loop forever so the generator never terminates
        
        # Shuffle the driving log to avoid very similar input
        driving_log = shuffle(driving_log)
        
        # Get a batch of driving log entries
        for offset in range(0, len(driving_log), batch_size):
            driving_log_batch = driving_log.iloc[offset:offset+batch_size]

            camera_images = []
            steering_angles = []
            
            # Loop through driving log entries and crete perturbed data for training the model
            for current_log_entry in zip(driving_log_batch['camera_image'], driving_log_batch['steering_angle']):
                
                # Read image (in RGB colorspace)
                current_image = cv2.imread(current_log_entry[0])
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                # Read steering angle
                current_steering_angle = float(current_log_entry[1])
                
                ### Image pertubations
                current_image, current_steering_angle = apply_random_flip(current_image, current_steering_angle)
                current_image = apply_random_brightness(current_image)
                current_image, current_steering_angle = apply_random_translation(current_image, current_steering_angle)
                
                # Add image to batch
                camera_images.append(current_image)
                steering_angles.append(current_steering_angle)

            # Convert data to arrays, as expected by KERAS            
            yield np.array(camera_images), np.array(steering_angles)

def main(_):
    ### Create generators
    # Split dataset
    training_log, validation_log = train_test_split(driving_log, test_size=0.2, shuffle=True)

    training_data_generator = get_batch(training_log, FLAGS.batch_size)
    validation_data_generator = get_batch(validation_log, FLAGS.batch_size)

    ### Train model
    print("Training model for {} epochs. Batch size = {}, lr = {}".format(FLAGS.epochs, FLAGS.batch_size,
                                                                          FLAGS.learning_rate))

    training_history = model.fit_generator(training_data_generator, samples_per_epoch=len(training_log), 
                                           validation_data=validation_data_generator, nb_val_samples=len(validation_log), 
                                           nb_epoch=FLAGS.epochs)


     #----- 3.3 Save trained model

    model.save("model.h5")

    #----- 3.4 Evaluate model

    ### Plot loss history graph
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])

    plt.title("Model Training History")
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training set", "Validation set"], loc="upper right")

    plt.savefig("./report_images/loss-history.png", bbox_inches='tight')


# --- MAIN ---
if __name__ == '__main__':
    tf.app.run()
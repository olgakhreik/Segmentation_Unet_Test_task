from constants import mask_csv, train_image_dir
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import make_image_gen, create_aug_gen
from keras import models, layers
import keras.backend as K
from keras.metrics import MeanIoU


masks = pd.read_csv(mask_csv, header=0)

# Add a column 'ships' to indicate if a row has an encoded pixel value or not
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

# Calculate the sum of 'ships' for each unique image and create additional columns
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

# some files are too small/corrupt / keep only 50kb files
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id:
                                                               os.stat(os.path.join(train_image_dir,
                                                                                    c_img_id)).st_size/1024)
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50]

# Drop the 'ships' column
masks.drop(['ships'], axis=1, inplace=True)

# Split the unique image IDs into training and validation sets
train_ids, valid_ids = train_test_split(unique_img_ids,
                 test_size = 0.2,
                 stratify = unique_img_ids['ships'])

# Merge the masks DataFrame with the training and validation IDs
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

# Create a new column 'grouped_ship_count' in the training DataFrame
train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x + 1) // 2).clip(0, 7)


def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].values[0] == 0:
        return in_df.sample(base_rep_val // 3)  # even more strongly undersample no ships
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0] < base_rep_val))


# Sample ships from the training DataFrame to balance the data
balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)

# Generate training and validation data using the balanced training DataFrame
train_gen = make_image_gen(balanced_train_df)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(make_image_gen(valid_df, 400))

# Create an augmented generator for training data
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)

#keep first 9 samples to examine in detail
t_x = t_x[:9]
t_y = t_y[:9]

# Define Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice

# Build U-Net model
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


upsample = upsample_simple

input_img = layers.Input(t_x.shape[1:], name='RGB_Input')
pp_in_layer = input_img

pp_in_layer = layers.GaussianNoise(0.1)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

u6 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

u7 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

u8 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

u9 = upsample(8, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
d = layers.Cropping2D((16, 16))(d)
d = layers.ZeroPadding2D((16, 16))(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])

seg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient, MeanIoU(num_classes=2)])

# Train the model
seg_model.fit(train_x, train_y, batch_size=4, epochs=10, validation_data=(valid_x, valid_y))

#save pretrained model
seg_model.save("seg_model.h5")

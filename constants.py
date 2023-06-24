import os


IMG_SCALING = (1, 1)
DATA_FOLDER = 'data'
train_image_dir = os.path.join(DATA_FOLDER, 'train_v2')
test_image_dir = os.path.join(DATA_FOLDER, 'test_v2')
mask_csv = os.path.join(DATA_FOLDER, 'train_ship_segmentations_v2.csv')
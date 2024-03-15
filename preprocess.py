import os

import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

num_workers = 0

batch_size = 20

validation_percentage = 0.2

train_file_list = []
test_file_list = []

HAS_WEAPON = f".\\Data\\HasWeapon"
NO_WEAPON = f".\\Data\\NoWeapon"


test_path = "test.jpg"
preprocessed_output_dir = "preprocessed_img"

LABELS = [HAS_WEAPON, NO_WEAPON]

def perform_img_augmentation(img, augmentation_num: int = 1) -> list:
    augmented_imgs = []

    aug_rotate = iaa.Sequential([
         # 1. Flip
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # 2. Affine
        iaa.Affine(
                rotate=(-30, 30),
                ),
        # 3. Multiply
        iaa.Multiply((0.8, 1.2)),
        # 4. Linearcontrast
        iaa.LinearContrast((0.6, 1.4)),
        # Perform methods below only sometimes
        iaa.Sometimes(0.5,
            # 5. GaussianBlur
            iaa.GaussianBlur((0.0, 3.0))
        )
    ])

    for i in range(augmentation_num):
        augmented_img = aug_rotate(images=[img])
        for aug_img in augmented_img:
            augmented_imgs.append(aug_img)
    return augmented_imgs

def write_all_augmentation(aug_img_list:list, path_prefix:str, img_size:int) -> None:
    for idx, img in enumerate(aug_img_list):
        img_to_write = cv2.resize(img, (img_size, img_size))
        write_path_augmented = path_prefix + str(idx) + ".jpg"
        cv2.imwrite(write_path_augmented, img_to_write)

# preprocess and load data from disk
def preprocess_data():
    corrupted_data_count = 0
    total_img_count = 0
    IMG_SIZE = 100
    
    for label in LABELS:
        class_name = os.path.basename(label)
        output_class_dir = os.path.join(preprocessed_output_dir, class_name)
        for file in tqdm(os.listdir(label), desc=f"preprocessing {label}"):
            total_img_count += 1
            try:
                if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
                    img_path = os.path.join(label, file)
                    img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE
                    augmented_img_list = perform_img_augmentation(img, 15)

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    os.makedirs(output_class_dir, exist_ok=True)
                    write_path = os.path.join(output_class_dir, file)
                    write_path_augmented_prefix = os.path.join(output_class_dir, f"{file}_augmented")

                    cv2.imwrite(write_path, img)
                    write_all_augmentation(augmented_img_list, write_path_augmented_prefix, IMG_SIZE)
                    
            except Exception as e:
                corrupted_data_count += 1
    print(f"Data preprocessed complete, {corrupted_data_count} of {total_img_count} image corrupted")

if __name__ == "__main__":
    print("Preprocessing")
    preprocess_data()
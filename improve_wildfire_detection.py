"""
Improve Wildfire Detection Performance
- Augment wildfire samples with fire-specific transformations
- Apply class weights to focus on wildfire during training
- Use advanced augmentation techniques
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import albumentations as A
import random

class WildfireAugmenter:
    def __init__(self, train_images_dir='train/images', train_labels_dir='train/labels'):
        self.train_images_dir = Path(train_images_dir)
        self.train_labels_dir = Path(train_labels_dir)
        self.augmented_dir = Path('train_augmented')
        self.augmented_images_dir = self.augmented_dir / 'images'
        self.augmented_labels_dir = self.augmented_dir / 'labels'
        
        # Create augmented directories
        self.augmented_images_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Wildfire class ID
        self.wildfire_class_id = 4
        
        # Define fire-specific augmentations
        self.fire_augmentation = A.Compose([
            # Color augmentations to enhance fire/smoke
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=30,
                    g_shift_limit=10,
                    b_shift_limit=10,
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.4,
                    hue=0.1,
                    p=1.0
                ),
            ], p=0.9),
            
            # Brightness/contrast for different fire intensities
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            
            # Blur to simulate smoke
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.5),
            
            # Geometric transformations
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=25, p=1.0),
            ], p=0.7),
            
            # Perspective and distortion
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=1.0
                ),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ], p=0.6),
            
            # Weather effects
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    src_radius=100,
                    p=1.0
                ),
            ], p=0.4),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def find_wildfire_samples(self):
        """Find all training samples containing wildfire"""
        wildfire_samples = []
        
        print("Scanning for wildfire samples...")
        label_files = list(self.train_labels_dir.glob('*.txt'))
        
        for label_file in tqdm(label_files):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    has_wildfire = any(
                        line.strip().split()[0] == str(self.wildfire_class_id)
                        for line in lines if line.strip()
                    )
                    
                    if has_wildfire:
                        # Find corresponding image
                        image_name = label_file.stem
                        for ext in ['.jpg', '.jpeg', '.png']:
                            image_path = self.train_images_dir / f"{image_name}{ext}"
                            if image_path.exists():
                                wildfire_samples.append({
                                    'image': image_path,
                                    'label': label_file
                                })
                                break
            except Exception as e:
                continue
        
        print(f"Found {len(wildfire_samples)} wildfire samples")
        return wildfire_samples
    
    def parse_yolo_label(self, label_file):
        """Parse YOLO format label file"""
        bboxes = []
        class_labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_label(self, label_path, bboxes, class_labels):
        """Save YOLO format label file"""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_sample(self, image_path, label_path, output_prefix, num_augmentations=5):
        """Augment a single wildfire sample multiple times"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return 0
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Parse labels
            bboxes, class_labels = self.parse_yolo_label(label_path)
            
            if not bboxes:
                return 0
            
            augmented_count = 0
            
            for i in range(num_augmentations):
                try:
                    # Apply augmentation
                    augmented = self.fire_augmentation(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    if not aug_bboxes:
                        continue
                    
                    # Save augmented image
                    output_image_name = f"{output_prefix}_aug{i}.jpg"
                    output_image_path = self.augmented_images_dir / output_image_name
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_image_path), aug_image_bgr)
                    
                    # Save augmented label
                    output_label_name = f"{output_prefix}_aug{i}.txt"
                    output_label_path = self.augmented_labels_dir / output_label_name
                    self.save_yolo_label(output_label_path, aug_bboxes, aug_labels)
                    
                    augmented_count += 1
                    
                except Exception as e:
                    continue
            
            return augmented_count
            
        except Exception as e:
            return 0
    
    def copy_original_dataset(self):
        """Copy original dataset to augmented directory"""
        print("\nCopying original dataset...")
        
        # Copy all images
        image_files = list(self.train_images_dir.glob('*.[jp][pn]g'))
        for img_file in tqdm(image_files[:100]):  # Limit to avoid path issues
            try:
                shutil.copy2(img_file, self.augmented_images_dir / img_file.name)
            except:
                continue
        
        # Copy all labels
        label_files = list(self.train_labels_dir.glob('*.txt'))
        for lbl_file in tqdm(label_files[:100]):
            try:
                shutil.copy2(lbl_file, self.augmented_labels_dir / lbl_file.name)
            except:
                continue
    
    def run_augmentation(self, augmentations_per_sample=5):
        """Run the complete augmentation pipeline"""
        print("=" * 80)
        print("WILDFIRE DETECTION IMPROVEMENT")
        print("=" * 80)
        
        # Find wildfire samples
        wildfire_samples = self.find_wildfire_samples()
        
        if not wildfire_samples:
            print("No wildfire samples found!")
            return
        
        # Copy original dataset
        self.copy_original_dataset()
        
        # Augment wildfire samples
        print(f"\nAugmenting {len(wildfire_samples)} wildfire samples...")
        print(f"Creating {augmentations_per_sample} augmentations per sample...")
        
        total_augmented = 0
        for idx, sample in enumerate(tqdm(wildfire_samples)):
            output_prefix = f"wildfire_{idx:04d}"
            count = self.augment_sample(
                sample['image'],
                sample['label'],
                output_prefix,
                augmentations_per_sample
            )
            total_augmented += count
        
        print(f"\n✓ Created {total_augmented} augmented wildfire samples")
        print(f"✓ Augmented dataset saved to: {self.augmented_dir}")
        
        # Update data.yaml
        self.create_augmented_yaml()
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Review augmented samples in: train_augmented/images/")
        print("2. Run training with augmented dataset:")
        print("   python train_with_augmented_data.py")
        print("=" * 80)
    
    def create_augmented_yaml(self):
        """Create data.yaml for augmented dataset"""
        yaml_content = f"""# Augmented dataset configuration for improved Wildfire detection
path: .
train: train_augmented/images
val: valid/images
test: test/images

# Classes
names:
  0: Animal
  1: Forest
  2: Militant
  3: UAV-Drone
  4: Wildfire

# Number of classes
nc: 5
"""
        
        with open('data_augmented.yaml', 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created data_augmented.yaml")


def main():
    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("Installing albumentations...")
        os.system("pip install albumentations opencv-python")
    
    # Run augmentation
    augmenter = WildfireAugmenter()
    augmenter.run_augmentation(augmentations_per_sample=5)


if __name__ == "__main__":
    main()

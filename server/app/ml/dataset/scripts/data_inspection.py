"""
Data Inspection Script for Train_Augmented_Balanced Dataset

This script performs data inspection on the training dataset,
including directory structure, image counts, image statistics, and class distribution.
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_dataset_path():
    """Get the path to the Train_Augmented_Balanced dataset."""
    script_dir = Path(__file__).parent
    dataset_path = script_dir.parent / "Train_Augmented_Balanced"
    return dataset_path


def get_reports_path():
    """Get or create the reports directory."""
    script_dir = Path(__file__).parent
    reports_path = script_dir.parent / "reports"
    reports_path.mkdir(exist_ok=True)
    return reports_path


def count_images_per_class(dataset_path):
    """Count the number of images in each class folder."""
    class_counts = defaultdict(int)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return class_counts
    
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            image_count = sum(
                1 for file in class_folder.iterdir()
                if file.is_file() and file.suffix.lower() in image_extensions
            )
            class_counts[class_name] = image_count
    
    return class_counts


def get_image_statistics(dataset_path):
    """Get statistics about images in the dataset."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_sizes = []
    image_dimensions = []
    
    if not dataset_path.exists():
        return None, None
    
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            for image_file in class_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    try:
                        # Get file size
                        size_kb = image_file.stat().st_size / 1024
                        image_sizes.append(size_kb)
                        
                        # Get image dimensions
                        img = Image.open(image_file)
                        image_dimensions.append(img.size)  # (width, height)
                    except Exception as e:
                        print(f"⚠️  Error reading {image_file}: {e}")
    
    return image_sizes, image_dimensions


def print_inspection_report(dataset_path, output_file=None):
    """Print a comprehensive data inspection report."""
    def output(text=""):
        """Output to both console and file."""
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    output("\n" + "="*70)
    output("📊 DATA INSPECTION REPORT - Train_Augmented_Balanced")
    output("="*70)
    
    output(f"\n📁 Dataset Path: {dataset_path}")
    
    if not dataset_path.exists():
        output(f"❌ Dataset path does not exist!")
        return
    
    # Class distribution
    output("\n" + "-"*70)
    output("📈 CLASS DISTRIBUTION")
    output("-"*70)
    
    class_counts = count_images_per_class(dataset_path)
    
    if not class_counts:
        output("❌ No classes found in the dataset!")
        return
    
    total_images = sum(class_counts.values())
    output(f"Total Classes: {len(class_counts)}")
    output(f"Total Images: {total_images}")
    output()
    
    # Sort by count descending
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_classes:
        percentage = (count / total_images) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        output(f"{class_name:30} | {count:5} images ({percentage:5.2f}%) | {bar}")
    
    # Image statistics
    output("\n" + "-"*70)
    output("📷 IMAGE STATISTICS")
    output("-"*70)
    
    image_sizes, image_dimensions = get_image_statistics(dataset_path)
    
    if image_sizes and image_dimensions:
        output(f"\nFile Size Statistics (in KB):")
        output(f"  Min: {min(image_sizes):.2f} KB")
        output(f"  Max: {max(image_sizes):.2f} KB")
        output(f"  Mean: {np.mean(image_sizes):.2f} KB")
        output(f"  Std Dev: {np.std(image_sizes):.2f} KB")
        
        # Image dimensions
        widths = [dim[0] for dim in image_dimensions]
        heights = [dim[1] for dim in image_dimensions]
        
        output(f"\nImage Dimensions (in pixels):")
        output(f"  Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}")
        output(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}")
        
        # Most common dimensions
        dim_counter = Counter(image_dimensions)
        most_common_dims = dim_counter.most_common(5)
        
        output(f"\nMost Common Image Dimensions:")
        for dims, count in most_common_dims:
            percentage = (count / len(image_dimensions)) * 100
            output(f"  {dims[0]}x{dims[1]}: {count} images ({percentage:.2f}%)")
    else:
        output("❌ No images found or unable to read image statistics")
    
    # Directory structure
    output("\n" + "-"*70)
    output("📂 DIRECTORY STRUCTURE")
    output("-"*70)
    
    class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    output(f"Number of class folders: {len(class_folders)}")
    output("\nClass folders:")
    for folder in sorted(class_folders):
        output(f"  - {folder.name}")
    
    output("\n" + "="*70)
    output("✅ Data inspection complete!")
    output("="*70 + "\n")


def main():
    """Main function to run the data inspection."""
    dataset_path = get_dataset_path()
    reports_path = get_reports_path()
    
    # Create timestamped report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_path / f"data_inspection_{timestamp}.txt"
    
    # Run inspection and write to file
    with open(report_file, 'w', encoding='utf-8') as f:
        print_inspection_report(dataset_path, f)
    
    print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()

"""
Simple Data Analysis Script for Train_Augmented_Balanced Dataset

Lightweight version focusing on basic statistics without heavy computations.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
from PIL import Image
import random


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


def count_classes_and_images(dataset_path):
    """Count classes and images quickly."""
    print("📊 Counting images...", end='', flush=True)
    
    class_counts = defaultdict(int)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            count = sum(
                1 for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            class_counts[class_folder.name] = count
    
    print(" ✅")
    return class_counts


def analyze_image_sizes(dataset_path, sample_size=100):
    """Analyze image sizes from a small sample."""
    print(f"📷 Analyzing {sample_size} random images...", end='', flush=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = []
    
    # Collect all images
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            for image_file in class_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    all_images.append(image_file)
    
    # Sample
    sample = random.sample(all_images, min(sample_size, len(all_images)))
    
    dimensions = []
    file_sizes = []
    
    for image_file in sample:
        try:
            img = Image.open(image_file)
            dimensions.append(img.size)
            file_sizes.append(image_file.stat().st_size / 1024)
        except:
            pass
    
    print(" ✅")
    return dimensions, file_sizes


def print_simple_analysis(dataset_path, output_file=None):
    """Print simple analysis report."""
    def output(text=""):
        """Output to console and file."""
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    output("\n" + "="*60)
    output("📊 SIMPLE DATA ANALYSIS REPORT")
    output("="*60)
    output(f"\n📁 Dataset: {dataset_path}")
    
    if not dataset_path.exists():
        output("❌ Dataset not found!")
        return
    
    # Count classes and images
    output("\n" + "-"*60)
    output("📈 CLASS INFORMATION")
    output("-"*60)
    
    class_counts = count_classes_and_images(dataset_path)
    total_images = sum(class_counts.values())
    
    output(f"\nTotal Classes: {len(class_counts)}")
    output(f"Total Images: {total_images}")
    output(f"Images per Class (avg): {total_images / len(class_counts):.1f}")
    
    # Show top/bottom classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    output("\nClass Distribution:")
    for class_name, count in sorted_classes:
        pct = (count / total_images) * 100
        bar = "█" * int(pct / 2)
        output(f"  {class_name:30} {count:5} images ({pct:5.1f}%) {bar}")
    
    # Check balance
    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    imbalance = max_class / min_class if min_class > 0 else 0
    
    output(f"\nBalance Status:")
    output(f"  Min class size: {min_class}")
    output(f"  Max class size: {max_class}")
    output(f"  Imbalance ratio: {imbalance:.2f}x")
    
    if imbalance < 1.5:
        output("  ✅ Dataset is WELL-BALANCED")
    elif imbalance < 3:
        output("  ⚠️  Dataset is MODERATELY BALANCED")
    else:
        output("  ❌ Dataset is IMBALANCED")
    
    # Analyze image sizes
    output("\n" + "-"*60)
    output("📷 IMAGE ANALYSIS (from sample)")
    output("-"*60)
    
    dimensions, file_sizes = analyze_image_sizes(dataset_path, sample_size=100)
    
    if dimensions:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        
        output(f"\nImage Dimensions (pixels):")
        output(f"  Width: {min(widths)} - {max(widths)} (avg: {np.mean(widths):.0f})")
        output(f"  Height: {min(heights)} - {max(heights)} (avg: {np.mean(heights):.0f})")
        
        # Most common size
        from collections import Counter
        dim_counts = Counter(dimensions)
        most_common = dim_counts.most_common(1)
        if most_common:
            output(f"  Most common: {most_common[0][0][0]}x{most_common[0][0][1]}")
    
    if file_sizes:
        output(f"\nFile Sizes (KB):")
        output(f"  Min: {min(file_sizes):.1f} KB")
        output(f"  Max: {max(file_sizes):.1f} KB")
        output(f"  Avg: {np.mean(file_sizes):.1f} KB")
    
    # Summary
    output("\n" + "-"*60)
    output("📋 SUMMARY")
    output("-"*60)
    
    output(f"\n✅ Dataset ready for training" if total_images > 0 else "❌ No images found")
    output(f"  Total trainable samples: {total_images}")
    output(f"  Unique classes: {len(class_counts)}")
    
    output("\n" + "="*60)
    output("✅ Analysis complete!")
    output("="*60 + "\n")


def main():
    """Main function."""
    dataset_path = get_dataset_path()
    reports_path = get_reports_path()
    
    # Create timestamped report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_path / f"data_analysis_{timestamp}.txt"
    
    # Run analysis
    with open(report_file, 'w', encoding='utf-8') as f:
        print_simple_analysis(dataset_path, f)
    
    print(f"📄 Report saved: {report_file}")


if __name__ == "__main__":
    main()

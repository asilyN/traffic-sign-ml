"""
Advanced Data Analysis Script for Train_Augmented_Balanced Dataset

This script performs deep statistical analysis and data profiling on the training dataset,
including correlation analysis, distribution analysis, augmentation impact analysis,
and comprehensive statistical reports.
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from PIL import Image
from scipy import stats
import random
import warnings
warnings.filterwarnings('ignore')

# Progress bar helper
def print_progress(current, total, prefix='', suffix='', length=40):
    """Print a simple progress bar."""
    if total == 0:
        return
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent:.1%} {suffix}', end='', flush=True)
    if current == total:
        print()  # New line when complete


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


def extract_image_features(dataset_path, sample_percentage=20):
    """Extract statistical features from all images (with sampling)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    features = {
        'brightness': [],
        'contrast': [],
        'saturation': [],
        'file_size': [],
        'aspect_ratio': [],
        'class_labels': [],
        'image_count_per_class': defaultdict(int)
    }
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return features
    
    # First pass: collect all image files
    all_images = []
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            for image_file in class_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    all_images.append((class_folder.name, image_file))
    
    # Sample images
    sample_size = max(1, int(len(all_images) * sample_percentage / 100))
    sampled_images = random.sample(all_images, min(sample_size, len(all_images)))
    
    print(f"\n📊 Processing {len(sampled_images)} images ({sample_percentage}% sample of {len(all_images)} total)")
    
    # Process sampled images
    for idx, (class_name, image_file) in enumerate(sampled_images):
        print_progress(idx + 1, len(sampled_images), prefix='Analyzing', suffix=image_file.name[:30])
        
        try:
            # Load image
            img = Image.open(image_file)
            img_array = np.array(img)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Calculate brightness
            brightness = np.mean(img_array)
            features['brightness'].append(brightness)
            
            # Calculate contrast (std of pixel values)
            contrast = np.std(img_array)
            features['contrast'].append(contrast)
            
            # Calculate saturation for RGB images
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                hsv = np.array(Image.fromarray(np.uint8(img_array)).convert('HSV'))
                saturation = np.mean(hsv[:, :, 1])
                features['saturation'].append(saturation)
            
            # File size in KB
            file_size = image_file.stat().st_size / 1024
            features['file_size'].append(file_size)
            
            # Aspect ratio
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 0
            features['aspect_ratio'].append(aspect_ratio)
            
            # Class label
            features['class_labels'].append(class_name)
            features['image_count_per_class'][class_name] += 1
            
        except Exception as e:
            pass  # Skip problematic images silently
    
    return features


def calculate_statistical_analysis(features):
    """Calculate detailed statistical analysis of features."""
    analysis = {}
    
    for feature_name in ['brightness', 'contrast', 'saturation', 'file_size', 'aspect_ratio']:
        if feature_name in features and len(features[feature_name]) > 0:
            data = np.array(features[feature_name])
            
            analysis[feature_name] = {
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std_dev': float(np.std(data)),
                'variance': float(np.var(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
    
    return analysis


def analyze_class_distribution(class_counts):
    """Analyze class distribution statistics."""
    counts = list(class_counts.values())
    
    distribution_stats = {
        'total_classes': len(class_counts),
        'total_samples': sum(counts),
        'class_mean': float(np.mean(counts)),
        'class_median': float(np.median(counts)),
        'class_std': float(np.std(counts)),
        'class_min': int(min(counts)),
        'class_max': int(max(counts)),
        'imbalance_ratio': float(max(counts) / min(counts)) if min(counts) > 0 else float('inf'),
        'dominant_class': max(class_counts, key=class_counts.get),
        'minority_class': min(class_counts, key=class_counts.get),
        'classes_list': class_counts
    }
    
    return distribution_stats


def detect_outliers(features, feature_name):
    """Detect outliers using IQR method."""
    if feature_name not in features or len(features[feature_name]) == 0:
        return None
    
    data = np.array(features[feature_name])
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = {
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_count': int(np.sum((data < lower_bound) | (data > upper_bound))),
        'outlier_percentage': float(np.sum((data < lower_bound) | (data > upper_bound)) / len(data) * 100)
    }
    
    return outliers


def analyze_data_augmentation_impact(dataset_path):
    """Analyze the impact of data augmentation on image properties."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    augmentation_indicators = {
        'rotated': 0,
        'flipped': 0,
        'brightness_adjusted': 0,
        'possibly_augmented': 0,
        'total_checked': 0
    }
    
    if not dataset_path.exists():
        return augmentation_indicators
    
    # Collect files first
    all_files = []
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            for image_file in class_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    all_files.append(image_file)
    
    # Process with progress
    for idx, image_file in enumerate(all_files):
        if idx % 100 == 0:  # Update progress every 100 files
            print_progress(idx, len(all_files), prefix='Checking augmentation', suffix=f'{idx}/{len(all_files)}')
        
        augmentation_indicators['total_checked'] += 1
        filename = image_file.stem.lower()
        
        if any(x in filename for x in ['rotate', 'rot', '_r_', 'flipped', 'flip', '_f_']):
            augmentation_indicators['possibly_augmented'] += 1
        if any(x in filename for x in ['rotate', 'rot', '_r_']):
            augmentation_indicators['rotated'] += 1
        if any(x in filename for x in ['flip', '_f_']):
            augmentation_indicators['flipped'] += 1
        if any(x in filename for x in ['bright', 'contrast', '_b_']):
            augmentation_indicators['brightness_adjusted'] += 1
    
    print_progress(len(all_files), len(all_files), prefix='Checking augmentation')
    return augmentation_indicators


def print_analysis_report(dataset_path, output_file=None):
    """Print comprehensive data analysis report."""
    def output(text=""):
        """Output to both console and file."""
        print(text)
        if output_file:
            output_file.write(text + "\n")
    
    output("\n" + "="*70)
    output("📊 ADVANCED DATA ANALYSIS REPORT")
    output("="*70)
    
    output(f"\n📁 Dataset Path: {dataset_path}")
    
    if not dataset_path.exists():
        output(f"❌ Dataset path does not exist!")
        return
    
    output("\n⚡ Using 20% sampling for faster analysis")
    
    # Extract features
    output("\n⏳ Extracting image features...")
    features = extract_image_features(dataset_path, sample_percentage=20)
    
    if not features['brightness']:
        output("❌ No images found in dataset!")
        return
    
    # Feature Statistics
    output("\n" + "-"*70)
    output("📈 IMAGE FEATURE STATISTICS")
    output("-"*70)
    
    stats_analysis = calculate_statistical_analysis(features)
    
    feature_order = ['brightness', 'contrast', 'saturation', 'file_size', 'aspect_ratio']
    for feature_name in feature_order:
        if feature_name in stats_analysis:
            output(f"\n{feature_name.upper()}:")
            stat_dict = stats_analysis[feature_name]
            output(f"  Mean: {stat_dict['mean']:.2f}")
            output(f"  Median: {stat_dict['median']:.2f}")
            output(f"  Std Dev: {stat_dict['std_dev']:.2f}")
            output(f"  Range: {stat_dict['min']:.2f} - {stat_dict['max']:.2f}")
            output(f"  Q1-Q3: {stat_dict['q25']:.2f} - {stat_dict['q75']:.2f}")
            output(f"  Skewness: {stat_dict['skewness']:.2f}")
            output(f"  Kurtosis: {stat_dict['kurtosis']:.2f}")
    
    # Class Distribution Analysis
    output("\n" + "-"*70)
    output("⚖️  CLASS DISTRIBUTION ANALYSIS")
    output("-"*70)
    
    class_dist = analyze_class_distribution(features['image_count_per_class'])
    output(f"\nTotal Classes: {class_dist['total_classes']}")
    output(f"Total Samples: {class_dist['total_samples']}")
    output(f"Samples per Class:")
    output(f"  Mean: {class_dist['class_mean']:.2f}")
    output(f"  Median: {class_dist['class_median']:.2f}")
    output(f"  Std Dev: {class_dist['class_std']:.2f}")
    output(f"  Range: {class_dist['class_min']} - {class_dist['class_max']}")
    output(f"  Imbalance Ratio: {class_dist['imbalance_ratio']:.2f}x")
    output(f"  Dominant Class: {class_dist['dominant_class']} ({features['image_count_per_class'][class_dist['dominant_class']]} images)")
    output(f"  Minority Class: {class_dist['minority_class']} ({features['image_count_per_class'][class_dist['minority_class']]} images)")
    
    # Outlier Detection
    output("\n" + "-"*70)
    output("🔍 OUTLIER DETECTION")
    output("-"*70)
    
    for feature_name in ['brightness', 'contrast', 'file_size']:
        outliers = detect_outliers(features, feature_name)
        if outliers:
            output(f"\n{feature_name.upper()}:")
            output(f"  Valid Range: {outliers['lower_bound']:.2f} - {outliers['upper_bound']:.2f}")
            output(f"  Outliers Found: {outliers['outlier_count']} ({outliers['outlier_percentage']:.2f}%)")
    
    # Data Augmentation Analysis
    output("\n" + "-"*70)
    output("🔄 DATA AUGMENTATION ANALYSIS")
    output("-"*70)
    
    aug_analysis = analyze_data_augmentation_impact(dataset_path)
    if aug_analysis['total_checked'] > 0:
        aug_percentage = (aug_analysis['possibly_augmented'] / aug_analysis['total_checked']) * 100
        output(f"\nTotal Images Checked: {aug_analysis['total_checked']}")
        output(f"Possibly Augmented: {aug_analysis['possibly_augmented']} ({aug_percentage:.2f}%)")
        output(f"  - Rotated patterns: {aug_analysis['rotated']}")
        output(f"  - Flipped patterns: {aug_analysis['flipped']}")
        output(f"  - Brightness adjusted: {aug_analysis['brightness_adjusted']}")
    
    # Data Quality Score
    output("\n" + "-"*70)
    output("📊 DATA QUALITY SCORE")
    output("-"*70)
    
    # Calculate quality metrics
    quality_score = 100
    
    # Check balance
    if class_dist['imbalance_ratio'] > 2:
        quality_score -= 15
    
    # Check for outliers
    for feature_name in ['brightness', 'contrast', 'file_size']:
        outliers = detect_outliers(features, feature_name)
        if outliers and outliers['outlier_percentage'] > 5:
            quality_score -= 5
    
    output(f"\nOverall Data Quality Score: {quality_score}/100")
    if quality_score >= 80:
        output("✅ EXCELLENT - Dataset quality is very good")
    elif quality_score >= 60:
        output("⚠️  GOOD - Dataset quality is acceptable")
    elif quality_score >= 40:
        output("⚠️  FAIR - Dataset may need some cleaning/balancing")
    else:
        output("❌ POOR - Dataset requires significant improvements")
    
    # Summary recommendations
    output("\n" + "-"*70)
    output("💡 RECOMMENDATIONS")
    output("-"*70)
    
    recommendations = []
    
    if class_dist['imbalance_ratio'] > 3:
        recommendations.append("• Consider further balancing classes - high imbalance ratio detected")
    
    if aug_percentage and aug_percentage < 30:
        recommendations.append("• Consider applying more augmentation to increase dataset diversity")
    elif aug_percentage and aug_percentage > 80:
        recommendations.append("• Dataset appears to be heavily augmented - may have synthetic patterns")
    
    brightness_outliers = detect_outliers(features, 'brightness')
    if brightness_outliers and brightness_outliers['outlier_percentage'] > 5:
        recommendations.append("• Found anomalous brightness values - consider preprocessing")
    
    if recommendations:
        for rec in recommendations:
            output(rec)
    else:
        output("✅ Dataset appears well-prepared for training")
    
    output("\n" + "="*70)
    output("✅ Data analysis complete!")
    output("="*70 + "\n")


def main():
    """Main function to run the data analysis."""
    dataset_path = get_dataset_path()
    reports_path = get_reports_path()
    
    # Create timestamped report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_path / f"data_analysis_{timestamp}.txt"
    
    # Run analysis and write to file
    with open(report_file, 'w', encoding='utf-8') as f:
        print_analysis_report(dataset_path, f)
    
    print(f"\n📄 Report saved to: {report_file}")


if __name__ == "__main__":
    main()

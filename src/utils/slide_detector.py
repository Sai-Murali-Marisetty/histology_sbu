#!/usr/bin/env python3
"""
slide_detector.py - Auto-detect slide type from filename

Identifies slide type (H&E or IHC marker) based on filename patterns.
"""

import re
from pathlib import Path


# Known IHC markers
IHC_MARKERS = {
    'CD3': 'T-cell marker',
    'GFAP': 'Astrocyte marker (glial fibrillary acidic protein)',
    'IBA1': 'Microglia marker (ionized calcium-binding adapter molecule 1)',
    'NF': 'Neurofilament',
    'PGP': 'Pan-neuronal marker (protein gene product 9.5)',
    'PGP9-5': 'Pan-neuronal marker (protein gene product 9.5)',
}


def detect_slide_type(slide_name):
    """
    Auto-detect slide type from filename.
    
    Args:
        slide_name: Slide filename (e.g., "CD3-S25.svs" or "HE-B17.svs")
    
    Returns:
        str: Slide type ('H&E', 'IHC_CD3', 'IHC_GFAP', etc.)
    
    Examples:
        >>> detect_slide_type("CD3-S25.svs")
        'IHC_CD3'
        >>> detect_slide_type("HE-B17.svs")
        'H&E'
        >>> detect_slide_type("GFAP-B27.svs")
        'IHC_GFAP'
    """
    # Remove extension and convert to uppercase
    slide_name = Path(slide_name).stem.upper()
    
    # Check for H&E
    if re.search(r'\bHE[-=_]', slide_name):
        return 'H&E'
    
    # Check for each IHC marker
    for marker in IHC_MARKERS.keys():
        if re.search(rf'\b{marker}[-_]', slide_name):
            return f'IHC_{marker.replace("-", "")}'
    
    # Check for PID/BIDS (assume H&E)
    if re.search(r'\b(PID|BIDS)\d+', slide_name):
        return 'H&E'
    
    # Unknown - default to H&E
    return 'H&E'


def is_ihc(slide_name):
    """Check if slide is IHC"""
    return detect_slide_type(slide_name).startswith('IHC_')


def get_marker(slide_name):
    """
    Get IHC marker name if applicable.
    
    Returns:
        str or None: Marker name (e.g., 'CD3') or None if H&E
    """
    slide_type = detect_slide_type(slide_name)
    if slide_type.startswith('IHC_'):
        return slide_type.split('_')[1]
    return None


def get_marker_info(marker):
    """Get description of marker"""
    marker = marker.upper().replace("-", "")
    for key, desc in IHC_MARKERS.items():
        if key.replace("-", "") == marker:
            return desc
    return "Unknown marker"


def batch_classify_slides(slide_dir):
    """
    Classify all slides in a directory.
    
    Args:
        slide_dir: Path to directory containing .svs files
    
    Returns:
        dict: {slide_name: slide_type}
    """
    from pathlib import Path
    
    slide_dir = Path(slide_dir)
    slides = list(slide_dir.glob("*.svs"))
    
    results = {}
    for slide_path in slides:
        slide_type = detect_slide_type(slide_path.name)
        results[slide_path.name] = slide_type
    
    return results


def print_classification_summary(slide_dir):
    """Print classification summary for all slides"""
    results = batch_classify_slides(slide_dir)
    
    # Group by type
    by_type = {}
    for slide, stype in results.items():
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(slide)
    
    print("\n" + "="*80)
    print("SLIDE CLASSIFICATION SUMMARY")
    print("="*80)
    
    for stype in sorted(by_type.keys()):
        slides = by_type[stype]
        print(f"\n{stype} ({len(slides)} slides):")
        for slide in sorted(slides):
            print(f"  - {slide}")
        
        # Add marker info for IHC
        if stype.startswith('IHC_'):
            marker = stype.split('_')[1]
            info = get_marker_info(marker)
            print(f"    ({info})")
    
    print("\n" + "="*80)
    print(f"Total: {len(results)} slides")
    print("="*80 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python slide_detector.py <slide_name>        # Detect single slide")
        print("  python slide_detector.py --batch <slide_dir> # Classify all slides")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("Error: Please provide slide directory")
            sys.exit(1)
        print_classification_summary(sys.argv[2])
    else:
        slide_name = sys.argv[1]
        slide_type = detect_slide_type(slide_name)
        print(f"{slide_name} → {slide_type}")
        
        if slide_type.startswith('IHC_'):
            marker = get_marker(slide_name)
            info = get_marker_info(marker)
            print(f"Marker: {marker} ({info})")

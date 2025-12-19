#!/usr/bin/env python3
"""
config_loader.py - Load and access slide configurations
"""

import yaml
from pathlib import Path


class SlideConfig:
    """Load and access slide type configurations"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'slide_config.yaml'
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_config(self, slide_type):
        """Get configuration for a slide type"""
        # Normalize slide type
        slide_type = slide_type.replace('_', '')  # IHC_CD3 -> IHCCD3
        
        # Try exact match first
        if slide_type in self.config:
            return self.config[slide_type]
        
        # Try with underscore
        slide_type_with_underscore = slide_type.replace('IHC', 'IHC_')
        if slide_type_with_underscore in self.config:
            return self.config[slide_type_with_underscore]
        
        # Fallback to default
        print(f"Warning: No config for '{slide_type}', using default")
        return self.config.get('default', {})
    
    def get_segmentation_params(self, slide_type):
        """Get segmentation parameters"""
        cfg = self.get_config(slide_type)
        return cfg.get('segmentation', {})
    
    def get_density_radii(self, slide_type):
        """Get density radii"""
        cfg = self.get_config(slide_type)
        return cfg.get('density_radii_um', [50, 100, 150])
    
    def get_brown_params(self, slide_type):
        """Get brown stain detection parameters"""
        cfg = self.get_config(slide_type)
        return cfg.get('brown_detection', {})
    
    def get_clustering_params(self, slide_type):
        """Get clustering parameters"""
        cfg = self.get_config(slide_type)
        return cfg.get('clustering', {})
    
    def needs_brown_stain(self, slide_type):
        """Check if brown stain analysis is needed"""
        cfg = self.get_config(slide_type)
        features = cfg.get('features', {})
        return features.get('brown_stain', False)


def load_config(config_path=None):
    """Quick helper to load config"""
    return SlideConfig(config_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_loader.py <slide_type>")
        print("Example: python config_loader.py IHC_CD3")
        sys.exit(1)
    
    slide_type = sys.argv[1]
    cfg = SlideConfig()
    
    print(f"\nConfiguration for {slide_type}:")
    print("="*60)
    
    config = cfg.get_config(slide_type)
    
    print(f"\nDescription: {config.get('description', 'N/A')}")
    
    print("\nSegmentation:")
    seg = config.get('segmentation', {})
    for key, val in seg.items():
        print(f"  {key}: {val}")
    
    print("\nDensity radii:", cfg.get_density_radii(slide_type))
    
    print("\nBrown stain needed:", cfg.needs_brown_stain(slide_type))
    if cfg.needs_brown_stain(slide_type):
        print("Brown detection params:")
        brown = cfg.get_brown_params(slide_type)
        for key, val in brown.items():
            print(f"  {key}: {val}")
    
    print("\nClustering:")
    clust = cfg.get_clustering_params(slide_type)
    print(f"  Features: {clust.get('features', [])}")
    print(f"  N clusters: {clust.get('n_clusters')}")

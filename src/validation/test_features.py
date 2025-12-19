#!/usr/bin/env python3
"""
Quick test to verify new features are working.
Run this on a sample CSV to check coherency and variance calculations.

Usage: python test_new_features.py --csv /path/to/nuclei_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def test_coherency_function():
    """Test coherency calculation on synthetic data"""
    print("\n" + "="*60)
    print("TEST 1: Coherency Calculation")
    print("="*60)
    
    # Create synthetic data with known alignment
    n = 100
    
    # Aligned nuclei (should give high coherency)
    aligned_df = pd.DataFrame({
        'x_um': np.linspace(0, 100, n),
        'y_um': np.linspace(0, 100, n),
        'major_axis_length': np.ones(n) * 20,
        'minor_axis_length': np.ones(n) * 10,  # All elongated same direction
    })
    
    # Random nuclei (should give low coherency)
    random_df = pd.DataFrame({
        'x_um': np.random.rand(n) * 100,
        'y_um': np.random.rand(n) * 100,
        'major_axis_length': np.random.rand(n) * 20 + 10,
        'minor_axis_length': np.random.rand(n) * 10 + 5,
    })
    
    # Import function
    import sys
    sys.path.insert(0, '.')
    from pathlib import Path
    
    # Load the coherency function
    import importlib.util
    spec = importlib.util.spec_from_file_location("enrich", "05_enrich_and_visualize_features.py")
    enrich = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enrich)
    
    aligned_coh = enrich.compute_coherency(aligned_df, 50.0)
    random_coh = enrich.compute_coherency(random_df, 50.0)
    
    print(f"Aligned nuclei coherency: {np.mean(aligned_coh):.3f} (expect ~0.8-1.0)")
    print(f"Random nuclei coherency: {np.mean(random_coh):.3f} (expect ~0.0-0.3)")
    
    if np.mean(aligned_coh) > 0.6 and np.mean(random_coh) < 0.4:
        print("✅ Coherency test PASSED")
        return True
    else:
        print("❌ Coherency test FAILED")
        return False


def test_variance_function():
    """Test variance calculation"""
    print("\n" + "="*60)
    print("TEST 2: Variance Calculation")
    print("="*60)
    
    # Create data with known variance
    n = 50
    
    # Uniform area (low variance)
    uniform_df = pd.DataFrame({
        'x_um': np.random.rand(n) * 100,
        'y_um': np.random.rand(n) * 100,
        'area_px': np.ones(n) * 100,  # All same size
    })
    
    # Variable area (high variance)
    variable_df = pd.DataFrame({
        'x_um': np.random.rand(n) * 100,
        'y_um': np.random.rand(n) * 100,
        'area_px': np.random.rand(n) * 200 + 50,  # Wide range
    })
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("enrich", "05_enrich_and_visualize_features.py")
    enrich = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enrich)
    
    _, uniform_var, _ = enrich.compute_local_stats(uniform_df, 'area_px', 50.0)
    _, variable_var, _ = enrich.compute_local_stats(variable_df, 'area_px', 50.0)
    
    print(f"Uniform area variance: {np.mean(uniform_var):.3f} (expect ~0)")
    print(f"Variable area variance: {np.mean(variable_var):.3f} (expect >100)")
    
    if np.mean(uniform_var) < 10 and np.mean(variable_var) > 100:
        print("✅ Variance test PASSED")
        return True
    else:
        print("❌ Variance test FAILED")
        return False


def check_csv_columns(csv_path):
    """Check if CSV has new columns"""
    print("\n" + "="*60)
    print("TEST 3: CSV Column Check")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    print(f"Total columns: {len(df.columns)}")
    print(f"Total nuclei: {len(df):,}")
    
    # Check for new columns
    expected_new = [
        'coherency_150um',
        'area_px_local_variance_150um',
        'aspect_ratio_local_variance_150um',
        'circularity_local_variance_150um'
    ]
    
    found = []
    missing = []
    
    for col in expected_new:
        if col in df.columns:
            found.append(col)
            print(f"  ✅ {col}")
            print(f"     Mean: {df[col].mean():.4f}, Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
        else:
            missing.append(col)
            print(f"  ❌ {col} - MISSING")
    
    print(f"\nFound {len(found)}/{len(expected_new)} expected columns")
    
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
        print("Re-run the pipeline with updated scripts!")
        return False
    else:
        print("\n✅ All expected columns present!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test new feature implementations")
    parser.add_argument("--csv", help="Path to enriched features CSV (optional)")
    parser.add_argument("--run_unit_tests", action="store_true", 
                       help="Run unit tests on synthetic data")
    args = parser.parse_args()
    
    results = []
    
    if args.run_unit_tests:
        results.append(("Coherency Test", test_coherency_function()))
        results.append(("Variance Test", test_variance_function()))
    
    if args.csv:
        if Path(args.csv).exists():
            results.append(("CSV Columns Check", check_csv_columns(args.csv)))
        else:
            print(f"Error: CSV not found: {args.csv}")
    
    if not args.run_unit_tests and not args.csv:
        print("Usage:")
        print("  python test_new_features.py --run_unit_tests")
        print("  python test_new_features.py --csv /path/to/features.csv")
        print("  python test_new_features.py --run_unit_tests --csv /path/to/features.csv")
        return
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        if all(r[1] for r in results):
            print("\n🎉 All tests passed! Features are working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check implementation.")


if __name__ == "__main__":
    main()

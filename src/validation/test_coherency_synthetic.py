#!/usr/bin/env python3
"""
Test coherency with REAL orientation angles in synthetic data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

def create_aligned_nuclei(n=100, base_angle=0):
    """All nuclei aligned in same direction"""
    np.random.seed(42)
    df = pd.DataFrame({
        'x_um': np.random.rand(n) * 1000,
        'y_um': np.random.rand(n) * 1000,
        'major_axis_length': np.random.rand(n) * 20 + 10,
        'minor_axis_length': np.random.rand(n) * 10 + 5,
        'orientation': np.ones(n) * base_angle + np.random.randn(n) * 0.05  # All same angle ± noise
    })
    return df

def create_random_nuclei(n=100):
    """Random orientations"""
    np.random.seed(43)
    df = pd.DataFrame({
        'x_um': np.random.rand(n) * 1000,
        'y_um': np.random.rand(n) * 1000,
        'major_axis_length': np.random.rand(n) * 20 + 10,
        'minor_axis_length': np.random.rand(n) * 10 + 5,
        'orientation': np.random.rand(n) * 2 * np.pi - np.pi  # Uniform random angles
    })
    return df

def create_two_groups(n=100):
    """Two groups with orthogonal orientations"""
    np.random.seed(44)
    n_half = n // 2
    
    # Group 1: horizontal (0 radians)
    g1 = pd.DataFrame({
        'x_um': np.random.rand(n_half) * 500,
        'y_um': np.random.rand(n_half) * 1000,
        'major_axis_length': np.ones(n_half) * 20,
        'minor_axis_length': np.ones(n_half) * 10,
        'orientation': np.zeros(n_half) + np.random.randn(n_half) * 0.1
    })
    
    # Group 2: vertical (π/2 radians)
    g2 = pd.DataFrame({
        'x_um': np.random.rand(n_half) * 500 + 500,
        'y_um': np.random.rand(n_half) * 1000,
        'major_axis_length': np.ones(n_half) * 20,
        'minor_axis_length': np.ones(n_half) * 10,
        'orientation': np.ones(n_half) * np.pi/2 + np.random.randn(n_half) * 0.1
    })
    
    return pd.concat([g1, g2], ignore_index=True)

def compute_coherency_FIXED(df, radius_um):
    """FIXED version using actual orientation"""
    from scipy.spatial import cKDTree
    
    coords = np.vstack([df["x_um"], df["y_um"]]).T
    tree = cKDTree(coords)
    
    # USE ACTUAL ORIENTATION
    angles = df['orientation'].to_numpy()
    
    coherency = []
    for i, p in enumerate(coords):
        idx = tree.query_ball_point(p, radius_um)
        
        if len(idx) < 3:
            coherency.append(0.0)
            continue
        
        local_angles = angles[idx]
        
        # Structure tensor
        Jxx = np.mean(np.cos(local_angles)**2)
        Jyy = np.mean(np.sin(local_angles)**2)
        Jxy = np.mean(np.cos(local_angles) * np.sin(local_angles))
        
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy**2
        
        discriminant = trace**2 - 4*det
        if discriminant < 0:
            discriminant = 0
        
        lambda1 = 0.5 * (trace + np.sqrt(discriminant))
        lambda2 = 0.5 * (trace - np.sqrt(discriminant))
        
        if lambda1 + lambda2 > 1e-10:
            coh = (lambda1 - lambda2) / (lambda1 + lambda2)
        else:
            coh = 0.0
        
        coherency.append(coh)
    
    return np.array(coherency)

def plot_test_case(df, title, coherency_vals):
    """Visualize with orientation vectors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: orientation vectors using ACTUAL orientation
    for idx, row in df.iterrows():
        angle = row['orientation']
        length = row['major_axis_length']
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        ax1.arrow(row['x_um'], row['y_um'], dx, dy,
                 head_width=5, head_length=8, fc='blue', ec='blue', alpha=0.6)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 1000)
    ax1.set_aspect('equal')
    ax1.set_title(f'{title}\nOrientation Vectors')
    
    # Right: coherency values
    sc = ax2.scatter(df['x_um'], df['y_um'], c=coherency_vals, 
                    cmap='RdYlGn', vmin=0, vmax=1, s=50)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1000)
    ax2.set_aspect('equal')
    ax2.set_title(f'Coherency\nMean: {np.mean(coherency_vals):.3f} ± {np.std(coherency_vals):.3f}')
    plt.colorbar(sc, ax=ax2)
    
    return fig

if __name__ == "__main__":
    radius = 200
    
    print("="*60)
    print("FIXED COHERENCY VALIDATION (using real orientation)")
    print("="*60)
    
    # Test 1
    print("\nTest 1: Aligned (EXPECT: ~1.0)")
    df_aligned = create_aligned_nuclei(n=100)
    coh_aligned = compute_coherency_FIXED(df_aligned, radius)
    print(f"  Result: {np.mean(coh_aligned):.3f} ± {np.std(coh_aligned):.3f}")
    fig1 = plot_test_case(df_aligned, "Aligned", coh_aligned)
    fig1.savefig('test_FIXED_aligned.png', dpi=150)
    plt.close()
    
    # Test 2
    print("\nTest 2: Random (EXPECT: ~0.2-0.3)")
    df_random = create_random_nuclei(n=100)
    coh_random = compute_coherency_FIXED(df_random, radius)
    print(f"  Result: {np.mean(coh_random):.3f} ± {np.std(coh_random):.3f}")
    fig2 = plot_test_case(df_random, "Random", coh_random)
    fig2.savefig('test_FIXED_random.png', dpi=150)
    plt.close()
    
    # Test 3
    print("\nTest 3: Two orthogonal groups (EXPECT: varies)")
    df_two = create_two_groups(n=100)
    coh_two = compute_coherency_FIXED(df_two, radius)
    print(f"  Overall: {np.mean(coh_two):.3f} ± {np.std(coh_two):.3f}")
    print(f"  Left:  {np.mean(coh_two[:50]):.3f}")
    print(f"  Right: {np.mean(coh_two[50:]):.3f}")
    fig3 = plot_test_case(df_two, "Two Groups", coh_two)
    fig3.savefig('test_FIXED_two_groups.png', dpi=150)
    plt.close()
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Aligned:  {np.mean(coh_aligned):.3f} ✓" if np.mean(coh_aligned) > 0.95 else f"Aligned: {np.mean(coh_aligned):.3f} ✗")
    print(f"Random:   {np.mean(coh_random):.3f} ✓" if np.mean(coh_random) < 0.4 else f"Random: {np.mean(coh_random):.3f} ✗")
    print(f"Variance: {np.std(coh_two):.3f} ✓" if np.std(coh_two) > 0.1 else f"Variance: {np.std(coh_two):.3f} ✗")
    print("\nImages: test_FIXED_*.png")

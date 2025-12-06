import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import simulations_3d

def plot_3d_trajectories(photons, title="3D Photon Trajectories"):
    """
    Plot 3D trajectories of photons.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot slab boundaries
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    z0 = np.zeros_like(xx)
    z1 = np.ones_like(xx)
    
    ax.plot_surface(xx, yy, z0, alpha=0.2, color='gray')
    ax.plot_surface(xx, yy, z1, alpha=0.2, color='gray')
    
    count = 0
    for p in photons:
        # Limit number of lines to avoid clutter if N is large
        if count > 100: 
            break
            
        traj = np.array(p.trajectory)
        
        color = 'blue'
        if p.status == 'escaped_top':
            color = 'green'
        elif p.status == 'escaped_bottom':
            color = 'blue'
        elif p.status == 'absorbed':
            color = 'red'
            
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color=color, alpha=0.5, linewidth=0.8)
        count += 1
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Optical Depth)')
    ax.set_title(title)
    ax.set_zlim(-0.1, 1.1)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Escaped Top'),
        Line2D([0], [0], color='blue', lw=2, label='Escaped Bottom'),
        Line2D([0], [0], color='red', lw=2, label='Absorbed')
    ]
    ax.legend(handles=legend_elements)
    
    plt.savefig('plot_3d_trajectories.png', dpi=300)
    print("Saved plot_3d_trajectories.png")
    plt.close()

def plot_exit_heatmap(photons):
    """
    Plot heatmap of exit positions at the top surface.
    """
    x_exit = []
    y_exit = []
    
    for p in photons:
        if p.status == 'escaped_top':
            x_exit.append(p.x)
            y_exit.append(p.y)
            
    if not x_exit:
        print("No photons escaped top for heatmap.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist2d(x_exit, y_exit, bins=50, cmap='viridis')
    plt.colorbar(label='Photon Count')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Exit Position Heatmap (Top Surface)')
    plt.axis('equal')
    
    plt.savefig('plot_exit_heatmap.png', dpi=300)
    print("Saved plot_exit_heatmap.png")
    plt.close()

def plot_parameter_sweep_trends():
    """
    Plot trends from the CSV results.
    """
    try:
        df = pd.read_csv('monte_carlo_results_3d.csv')
    except FileNotFoundError:
        print("monte_carlo_results_3d.csv not found. Skipping sweep plots.")
        return

    # 1. Transmission vs Tau for different g
    plt.figure(figsize=(10, 6))
    
    # Filter for high albedo to see scattering effects clearly
    omega_val = 0.99
    subset = df[np.isclose(df['omega'], omega_val, atol=0.01)]
    
    unique_gs = sorted(subset['g'].unique())
    
    for g in unique_gs:
        data = subset[subset['g'] == g].sort_values('tau_tot')
        plt.plot(data['tau_tot'], data['escape_fraction_top'], 'o-', label=f'g={g}')
        
    plt.xscale('log')
    plt.xlabel('Optical Depth (Tau)')
    plt.ylabel('Transmission (Escape Top)')
    plt.title(f'Transmission vs Optical Depth (omega={omega_val})')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    
    plt.savefig('plot_transmission_vs_tau.png', dpi=300)
    print("Saved plot_transmission_vs_tau.png")
    plt.close()

    # 2. Conservation of Energy Check (Stacked Area)
    # Pick one g value
    g_val = 0.0
    subset_g = df[(df['g'] == g_val) & (np.isclose(df['omega'], 0.55, atol=0.01))].sort_values('tau_tot')
    
    if not subset_g.empty:
        plt.figure(figsize=(10, 6))
        
        x = subset_g['tau_tot']
        y1 = subset_g['escape_fraction_top']
        y2 = subset_g['escape_fraction_bottom']
        y3 = subset_g['absorbed_fraction']
        
        plt.stackplot(x, y1, y2, y3, labels=['Escape Top', 'Escape Bottom', 'Absorbed'], 
                      colors=['green', 'blue', 'red'], alpha=0.6)
        
        plt.xscale('log')
        plt.xlabel('Optical Depth (Tau)')
        plt.ylabel('Fraction')
        plt.title(f'Energy Conservation Check (g={g_val}, omega=0.5)')
        plt.legend(loc='center right')
        plt.ylim(0, 1.1)
        
        plt.savefig('plot_energy_conservation.png', dpi=300)
        print("Saved plot_energy_conservation.png")
        plt.close()
    else:
        print(f"No data found for g={g_val}, omega=0.55 for energy conservation plot.")

def plot_albedo_impact_heatmap():
    """
    Plot heatmap of Transmission vs Tau and Omega.
    """
    try:
        df = pd.read_csv('monte_carlo_results_3d.csv')
    except FileNotFoundError:
        return

    # Filter for a specific g (e.g., g=0)
    g_val = 0.0
    subset = df[df['g'] == g_val]
    
    if subset.empty:
        return

    # Pivot data for heatmap
    pivot_table = subset.pivot(index='omega', columns='tau_tot', values='escape_fraction_top')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_table, origin='lower', aspect='auto', cmap='magma',
               extent=[pivot_table.columns.min(), pivot_table.columns.max(), 
                       pivot_table.index.min(), pivot_table.index.max()])
    
    plt.colorbar(label='Transmission (Escape Top)')
    plt.xlabel('Optical Depth (Tau)')
    plt.ylabel('Albedo (Omega)')
    plt.title(f'Transmission Phase Space (g={g_val})')
    
    plt.savefig('plot_albedo_impact_heatmap.png', dpi=300)
    print("Saved plot_albedo_impact_heatmap.png")
    plt.close()

def main():
    print("Running detailed simulation for visualization...")
    # Run a single detailed simulation
    # tau=5 (moderately thick), omega=0.9 (scattering dominant), g=0.5 (forward scattering)
    photons = simulations_3d.run_detailed_mc_3d(tau_tot=5.0, omega=0.9, g=0.5, N=200)
    
    print(f"Simulation complete. {len(photons)} photons traced.")
    
    plot_3d_trajectories(photons)
    plot_exit_heatmap(photons)
    plot_parameter_sweep_trends()
    plot_albedo_impact_heatmap()
    
    print("All visualizations generated.")

if __name__ == "__main__":
    main()

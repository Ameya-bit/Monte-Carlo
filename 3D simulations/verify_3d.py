import pandas as pd
import numpy as np

def verify_results():
    print("Verifying 3D Monte Carlo Results...")
    
    # Load results
    try:
        df = pd.read_csv('monte_carlo_results_3d.csv')
    except FileNotFoundError:
        print("Error: monte_carlo_results_3d.csv not found.")
        return

    # 1. Check Conservation of Energy
    # Total probability should sum to 1.0 (within floating point error)
    df['total_prob'] = df['escape_fraction_top'] + df['escape_fraction_bottom'] + df['absorbed_fraction']
    
    # Check if any row deviates significantly from 1.0
    tolerance = 1e-5
    violations = df[np.abs(df['total_prob'] - 1.0) > tolerance]
    
    if len(violations) == 0:
        print("✅ Conservation of Energy: PASSED (All rows sum to 1.0)")
    else:
        print(f"❌ Conservation of Energy: FAILED ({len(violations)} violations found)")
        print(violations.head())

    # 2. Check Trends
    # As tau increases, transmission (escape_top) should decrease
    print("\nChecking Trends (Tau vs Escape Top):")
    # Filter for g=0, omega=1 (pure scattering)
    subset = df[(df['g'] == 0) & (df['omega'] > 0.98)]
    if not subset.empty:
        print(subset[['tau_tot', 'escape_fraction_top']].sort_values('tau_tot'))
    else:
        print("No pure scattering subset found.")

    # 3. Check Asymmetry Effect
    # For same tau/omega, higher g (forward scattering) should have higher transmission
    print("\nChecking Asymmetry Effect (g=0 vs g=0.9):")
    tau_check = 1.0
    omega_check = 0.99
    
    # Find closest values
    row_g0 = df[(df['g'] == 0) & (np.isclose(df['tau_tot'], tau_check, atol=0.1)) & (np.isclose(df['omega'], omega_check, atol=0.01))]
    row_g9 = df[(df['g'] == 0.9) & (np.isclose(df['tau_tot'], tau_check, atol=0.1)) & (np.isclose(df['omega'], omega_check, atol=0.01))]
    
    if not row_g0.empty and not row_g9.empty:
        t0 = row_g0.iloc[0]['escape_fraction_top']
        t9 = row_g9.iloc[0]['escape_fraction_top']
        print(f"Transmission at g=0:   {t0:.4f}")
        print(f"Transmission at g=0.9: {t9:.4f}")
        
        if t9 > t0:
            print("✅ Forward scattering increases transmission: PASSED")
        else:
            print("❌ Forward scattering check: FAILED")
    else:
        print("Could not find comparable rows for asymmetry check.")

if __name__ == "__main__":
    verify_results()

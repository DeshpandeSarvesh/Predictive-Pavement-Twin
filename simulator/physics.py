import math

# =====================================================================
# CIVIL ENGINEERING PHYSICS ENGINE: IRC:37-2018 IMPLEMENTATION
# =====================================================================
# This script implements the empirical-mechanistic pavement design and
# degradation formulas dictated by the Indian Roads Congress (IRC:37).
# =====================================================================

def calculate_axle_equivalence(axle_type, load_kn):
    """
    Converts a specific axle configuration carrying a load in kN into 
    Equivalent Standard Axle repetitions (relative to an 80 kN standard axle).
    This implements IRC:37 Section 4.4.3 (The Fourth Power Law formulas).
    
    Parameters:
      - axle_type: 'single_wheel', 'single_dual', 'tandem_dual', 'tridem_dual'
      - load_kn: actual load carried by the axle in kilonewtons (kN)
    """
    if load_kn <= 0:
        return 0.0
        
    if axle_type == 'single_wheel':
        # Single axle with single wheel on either side (Standard denominator: 65 kN)
        return (load_kn / 65.0) ** 4
    elif axle_type == 'single_dual':
        # Single axle with dual wheels on either side (Standard denominator: 80 kN)
        return (load_kn / 80.0) ** 4
    elif axle_type == 'tandem_dual':
        # Tandem axle with dual wheels on either side (Standard denominator: 148 kN)
        return (load_kn / 148.0) ** 4
    elif axle_type == 'tridem_dual':
        # Tridem axle with dual wheels on either side (Standard denominator: 224 kN)
        return (load_kn / 224.0) ** 4
    else:
        # Fallback to standard dual wheel single axle
        return (load_kn / 80.0) ** 4

def calculate_design_traffic(A, D, F, n, r):
    """
    Calculates cumulative standard axle repetitions (N_des) over a design period.
    Implements IRC:37 Equation 4.6.
    
    Parameters:
      - A: Initial traffic in the year of completion (commercial vehicles per day)
      - D: Lateral distribution factor (e.g., 0.75 for two-lane undivided road)
      - F: Vehicle Damage Factor (VDF)
      - n: Design period in years
      - r: Annual growth rate in decimal (e.g., 0.06 for 6%)
    """
    if r <= 0:
        # Avoid division by zero
        return 365.0 * A * D * F * n
        
    numerator = 365.0 * ((1 + r) ** n - 1)
    N_des = (numerator / r) * A * D * F
    return N_des

def calculate_resilient_modulus_subgrade(cbr):
    """
    Calculates the Resilient Modulus of the Subgrade (M_RS) in MPa from its CBR value.
    Implements IRC:37 Section 6.3 Equations:
      - M_RS = 17.6 * CBR^0.64  (for CBR > 5%)
      - M_RS = 10.0 * CBR       (for CBR <= 5%)
    The maximum modulus is capped at 100 MPa as per Section 6.4.2.
    """
    if cbr <= 5.0:
        m_rs = 10.0 * cbr
    else:
        m_rs = 17.6 * (cbr ** 0.64)
        
    # Cap at 100 MPa as per design guidelines
    return min(m_rs, 100.0)

def calculate_pavement_temperature(air_temp, latitude):
    """
    Predicts the maximum pavement temperature at 20 mm depth.
    Implements IRC:37 Section A.5.5 Equation A-1.
    """
    term = (air_temp - 0.00618 * (latitude ** 2) + 0.2289 * latitude + 42.2) * 0.9545
    t_20mm = term - 17.78
    return t_20mm

def calculate_fatigue_life_nf(epsilon_t, mr_bit):
    """
    Calculates the allowable standard axle repetitions before fatigue cracking (N_f).
    Implements the Bituminous Fatigue performance model for 80% reliability:
    N_f = 0.5161 * 10^-4 * (1 / epsilon_t)^3.89 * (1 / mr_bit)^0.854
    """
    if epsilon_t <= 0 or mr_bit <= 0:
        return float('inf')
        
    term_strain = (1.0 / epsilon_t) ** 3.89
    term_modulus = (1.0 / mr_bit) ** 0.854
    n_f = 0.5161e-4 * term_strain * term_modulus
    return n_f

def calculate_rutting_life_nr(epsilon_v):
    """
    Calculates the allowable standard axle repetitions before subgrade rutting (N_r).
    Implements the Subgrade Rutting performance model for 80% reliability:
    N_r = 4.1656 * 10^-8 * (1 / epsilon_v)^4.5337
    """
    if epsilon_v <= 0:
        return float('inf')
        
    n_r = 4.1656e-8 * ((1.0 / epsilon_v) ** 4.5337)
    return n_r

def back_calculate_design_strains(design_life_msa, mr_bit, mr_sub):
    """
    Reverse-calculates the initial allowable strains (tensile strain at bottom of asphalt,
    and compressive strain at top of subgrade) matching the road's design class.
    This calibrates the baseline pavement model in structural equilibrium.
    """
    # Convert Million Standard Axles (MSA) to actual load repetitions
    repetitions = design_life_msa * 1e6
    
    # 1. Back-calculate allowable tensile strain (epsilon_t) from Fatigue Model
    # N_f = 0.5161e-4 * (1/eps_t)^3.89 * (1/mr_bit)^0.854
    # => (1/eps_t)^3.89 = N_f / [0.5161e-4 * (1/mr_bit)^0.854]
    # => eps_t = [ (0.5161e-4 * (1/mr_bit)^0.854) / N_f ] ^ (1 / 3.89)
    term_mod = (1.0 / mr_bit) ** 0.854
    epsilon_t = ((0.5161e-4 * term_mod) / repetitions) ** (1.0 / 3.89)
    
    # 2. Back-calculate allowable compressive strain (epsilon_v) from Rutting Model
    # N_r = 4.1656e-8 * (1/eps_v)^4.5337
    # => eps_v = [ 4.1656e-8 / N_r ] ^ (1 / 4.5337)
    epsilon_v = (4.1656e-8 / repetitions) ** (1.0 / 4.5337)
    
    return epsilon_t, epsilon_v

def simulate_monsoon_weakening(cbr_initial, drainage_quality, is_monsoon):
    """
    Simulates seasonal moisture infiltration and subgrade weakening.
    Calculates the effective CBR based on the moisture modifier 'm':
      - Dry Season: m = 1.0 (Full design strength)
      - Monsoon + Good Drainage: m = 0.8 (Minor water ingress)
      - Monsoon + Poor Drainage: m = 0.3 (Severe waterlogging / strength collapse)
    """
    if not is_monsoon:
        m = 1.0
    else:
        # Drainage quality ranges from 0.0 (terrible) to 1.0 (perfect)
        # We classify >= 0.75 as good drainage
        if drainage_quality >= 0.75:
            m = 0.8
        else:
            m = 0.3
            
    cbr_effective = cbr_initial * m
    return cbr_effective

def calculate_bpr_travel_time(free_flow_time, traffic_volume, capacity, pci):
    """
    Implements the Bureau of Public Roads (BPR) travel impedance formula,
    weighted by the pavement condition index (PCI).
    As PCI drops, roughness increases (higher IRI), forcing vehicles to decelerate.
    
    Travel Time = FreeFlowTime * [1 + 0.15 * (V/C)^4] + PotholeDeceleration
    """
    # Standard BPR congestion factor
    congestion_factor = 1.0 + 0.15 * ((traffic_volume / max(capacity, 1.0)) ** 4)
    base_time = free_flow_time * congestion_factor
    
    # Pothole and roughness deceleration factor (IRI correlation)
    # If PCI = 100, no deceleration. If PCI = 20, significant delay added (rough road)
    roughness_delay = 0.05 * (100.0 - pci)
    
    return base_time + roughness_delay

if __name__ == "__main__":
    # Test script: Validate the equations
    print("Testing IRC:37 Physics Engine Calculations...")
    
    # 1. Test resilient modulus calculation
    cbr_good = 8.0
    cbr_poor = 3.0
    mr_good = calculate_resilient_modulus_subgrade(cbr_good)
    mr_poor = calculate_resilient_modulus_subgrade(cbr_poor)
    print(f"Subgrade Modulus (CBR={cbr_good}%): {mr_good:.2f} MPa")
    print(f"Subgrade Modulus (CBR={cbr_poor}%): {mr_poor:.2f} MPa")
    
    # 2. Test strain back-calculation for a 10 MSA road
    mr_bit = 3000.0  # Bituminous layer modulus at 35°C (MPa)
    eps_t, eps_v = back_calculate_design_strains(design_life_msa=10.0, mr_bit=mr_bit, mr_sub=mr_good)
    print(f"For 10 MSA Design Life:")
    print(f"  - Allowable tensile strain (eps_t): {eps_t * 1e6:.2f} microstrain")
    print(f"  - Allowable subgrade strain (eps_v): {eps_v * 1e6:.2f} microstrain")
    
    # 3. Test fatigue and rutting life from strains
    nf = calculate_fatigue_life_nf(eps_t, mr_bit)
    nr = calculate_rutting_life_nr(eps_v)
    print(f"Calculated Fatigue Life: {nf/1e6:.2f} Million Axles")
    print(f"Calculated Rutting Life: {nr/1e6:.2f} Million Axles")

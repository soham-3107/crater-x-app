import numpy as np
from scipy.interpolate import interp1d

class SubsurfaceIceEstimator:
    def __init__(self, max_depth=5.0, dz=0.1, regolith_density=1.5):
        """
        max_depth: Maximum depth in meters (default 5.0m)
        dz: vertical step size in meters (default 0.1m, so 50 layers)
        regolith_density: Density in metric tons per cubic meter (default 1.5 t/m^3)
        """
        self.max_depth = max_depth
        self.dz = dz
        self.depths = np.arange(0, max_depth + dz, dz)
        self.regolith_density = regolith_density

    def simulate_thermal_profile(self, z, T_surface=40.0, T_interior=230.0, d_thermal=1.8):
        """
        Simulates the subsurface temperature profile at depth z (meters)
        based on lunar geothermal heat flux.
        T(z) = T_surface + (T_interior - T_surface) * (1 - e^(-z / d_thermal))
        """
        return T_surface + (T_interior - T_surface) * (1.0 - np.exp(-z / d_thermal))

    def calculate_ice_stability(self, T):
        """
        Calculates ice thermal stability fraction (0.0 to 1.0).
        Ice sublimates rapidly in vacuum above 110K.
        At < 100K it is stable for billions of years (fraction = 1.0).
        At > 140K it sublimates instantly (fraction = 0.0).
        """
        stability = np.zeros_like(T)
        
        # Linear transition from 100K (fully stable) to 140K (completely sublimated)
        mask_stable = T <= 100.0
        mask_transition = (T > 100.0) & (T < 140.0)
        
        stability[mask_stable] = 1.0
        stability[mask_transition] = 1.0 - (T[mask_transition] - 100.0) / 40.0
        
        return np.clip(stability, 0.0, 1.0)

    def estimate_3d_ice_grid(self, cpr, dop, resolution):
        """
        Estimates a 3D grid of ice concentration (weight percentage) up to max_depth.
        Shape: (H, W, DepthSteps)
        
        Using a physical model where ice is detected at the surface by CPR > 1.0 and DOP < 0.13,
        and its concentration vertically varies based on:
        1. Local thermal stability (subsurface temperature)
        2. Soil diffusion profile (modeled via SciPy interp1d/depletion curve)
        """
        h, w = cpr.shape
        num_layers = len(self.depths)
        
        # Initialize 3D matrices
        ice_3d = np.zeros((h, w, num_layers))
        temp_3d = np.zeros((h, w, num_layers))
        
        # 1. Surface ice presence indicator (using the show-stopper polarimetric criteria)
        # We also allow high-resolution continuous estimation:
        # Ice index goes up as CPR increases above 1.0, and DOP decreases below 0.13
        ice_probability = np.zeros((h, w))
        ice_mask = (cpr >= 1.0) & (dop <= 0.13)
        
        # Quantify ice weight percentage at the "richest" subsurface boundary
        # Typically, ice concentration varies between 1% to 15% in lunar regolith
        ice_probability[ice_mask] = (cpr[ice_mask] - 1.0) / 1.5 * (0.13 - dop[ice_mask]) / 0.13
        surface_ice_pct = np.clip(ice_probability * 15.0, 0.0, 15.0) # max 15 wt.%
        
        # 2. Compute vertical profile layer-by-layer
        for idx, z in enumerate(self.depths):
            # Base temperature for PSR surface is 40K. If not PSR, it can be higher
            # We assume where ice is detected, it is in a shaded cold-trap, so T_surface = 40K
            # In other non-ice locations, T_surface can be modeled as higher (e.g. 120K)
            T_surf = np.where(ice_mask, 40.0, 120.0)
            
            # Subsurface temperature profile at depth z
            T_z = self.simulate_thermal_profile(z, T_surface=T_surf)
            temp_3d[:, :, idx] = T_z
            
            # Thermal stability factor
            stability_z = self.calculate_ice_stability(T_z)
            
            # Volumetric distribution factor (vertical diffusion profile)
            # Typically, ice is depleted at the very top (0 - 0.2m) due to solar wind sputtering,
            # peaks at 1 - 2.5m, and decays at 4 - 5m due to geothermal heating.
            # We model this curve:
            if z < 0.2:
                # Sputtering depletion
                depth_factor = 0.2 + 0.8 * (z / 0.2)
            else:
                # Geothermal boundary decay
                depth_factor = np.exp(-(z - 0.2) / 3.0)
                
            # Final concentration: surface concentration * depth profile * thermal stability
            ice_3d[:, :, idx] = surface_ice_pct * depth_factor * stability_z
            
        return ice_3d, temp_3d

    def calculate_total_water_mass(self, ice_3d, resolution):
        """
        Calculates the total water ice mass in metric tons.
        Mass = Volume * Density * Concentration
        Each cell volume = resolution_x * resolution_y * dz
        Cell mass of water = volume * regolith_density * (ice_wt_pct / 100.0)
        """
        cell_volume = resolution * resolution * self.dz  # m^3
        # ice_3d contains wt.%, convert to fraction
        ice_fraction = ice_3d / 100.0
        
        # Total mass in metric tons
        total_mass = np.sum(ice_fraction) * cell_volume * self.regolith_density
        return total_mass

    def get_drill_core(self, y_idx, x_idx, ice_3d, temp_3d):
        """
        Retrieves a 1D vertical column profile for a specific pixel.
        """
        # Ensure indices are within bounds
        h, w, _ = ice_3d.shape
        y = np.clip(y_idx, 0, h - 1)
        x = np.clip(x_idx, 0, w - 1)
        
        ice_profile = ice_3d[y, x, :]
        temp_profile = temp_3d[y, x, :]
        
        return {
            "depths": self.depths,
            "ice_concentration": ice_profile,
            "temperature": temp_profile,
            "y": y,
            "x": x
        }

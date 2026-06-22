import numpy as np
import scipy.ndimage as ndimage
import cv2

# Try to import rasterio, fallback to simulated engine if not available
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

class LunarCraterSimulator:
    def __init__(self, size=150, resolution=10.0):
        """
        size: grid size (size x size pixels)
        resolution: resolution in meters per pixel (default 10m/px, total 1.5km x 1.5km terrain)
        """
        self.size = size
        self.resolution = resolution
        self.x = np.arange(size) - size // 2
        self.y = np.arange(size) - size // 2
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        
    def generate_crater(self, name="Crater-X", center_peak=False, rim_radius=45.0, depth=800.0, height_rim=150.0):
        """
        Generates simulated elevation model (DEM) for a lunar crater.
        """
        # Set seed based on crater name for reproducibility
        seed = sum(ord(c) for c in name)
        np.random.seed(seed)
        
        # Base terrain: gently sloping plain with low-frequency noise
        base_noise = self._generate_perlin_noise(octaves=3, scale=40.0) * 50.0
        dem = base_noise.copy()
        
        # Crater bowl profile (Gaussian-like depression)
        bowl_width = rim_radius * 0.95
        crater_bowl = -depth * np.exp(-(self.R**2) / (2 * (bowl_width**2)))
        
        # Crater rim (annular ring)
        rim_width = rim_radius * 0.2
        crater_rim = height_rim * np.exp(-((self.R - rim_radius)**2) / (2 * (rim_width**2)))
        
        # Combine
        dem += crater_bowl + crater_rim
        
        # Optional center peak (complex crater)
        if center_peak:
            peak_width = rim_radius * 0.12
            peak_height = depth * 0.25
            peak = peak_height * np.exp(-(self.R**2) / (2 * (peak_width**2)))
            dem += peak
            
        # Add fine-grained surface roughness (rocks, ejecta)
        roughness = self._generate_perlin_noise(octaves=5, scale=8.0) * 15.0
        dem += roughness
        
        # Calculate Slope (in degrees)
        dy, dx = np.gradient(dem, self.resolution)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180.0 / np.pi)
        
        # Generate raw polarimetric radar features
        cpr, dop = self._generate_polarimetric_data(dem, slope, rim_radius)
        
        # Generate rock hazard distribution
        rock_density = self._generate_rock_density(slope)
        
        return {
            "name": name,
            "dem": dem,
            "slope": slope,
            "cpr": cpr,
            "dop": dop,
            "rock_density": rock_density,
            "resolution": self.resolution
        }

    def _generate_perlin_noise(self, octaves=3, scale=20.0):
        """
        Generates fractal-like noise.
        """
        noise = np.zeros((self.size, self.size))
        for o in range(octaves):
            freq = 2**o
            amp = 0.5**o
            # Generate random small grid
            small_size = max(4, int(self.size / scale * freq))
            small_grid = np.random.randn(small_size, small_size)
            # Resize to full grid size using bilinear interpolation
            resized = cv2.resize(small_grid, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            noise += amp * resized
        return noise

    def _generate_polarimetric_data(self, dem, slope, rim_radius):
        """
        Generates simulated Circular Polarization Ratio (CPR) and Degree of Polarization (DOP).
        Ice yields CPR > 1.0, DOP < 0.13.
        Rocky/rough areas yield CPR > 1.0 but DOP > 0.4.
        """
        # Base regolith values
        cpr = 0.15 + 0.1 * np.random.randn(self.size, self.size)
        dop = 0.85 + 0.05 * np.random.randn(self.size, self.size)
        
        # Rocky slopes increase roughness (increases CPR, decreases DOP slightly)
        slope_normalized = np.clip(slope / 30.0, 0, 1)
        cpr += slope_normalized * 0.65
        dop -= slope_normalized * 0.25
        
        # Permanently Shadowed Region (PSR) simulation: center of crater
        # In a real polar crater, the bottom remains in shadow.
        # Let's place localized subsurface/surface water ice deposits in the deepest part of the crater
        # inside r < rim_radius * 0.6.
        psr_mask = (self.R < rim_radius * 0.55) & (dem < np.percentile(dem, 20))
        
        # Let's seed 3-4 distinct ice pockets inside the PSR
        np.random.seed(42)
        ice_pockets = np.zeros_like(dem, dtype=bool)
        for _ in range(4):
            cx = np.random.randint(-int(rim_radius*0.4), int(rim_radius*0.4))
            cy = np.random.randint(-int(rim_radius*0.4), int(rim_radius*0.4))
            r_ice = np.random.uniform(5.0, 15.0)
            pocket = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2) < r_ice
            ice_pockets = ice_pockets | pocket
            
        ice_mask = psr_mask & ice_pockets
        
        # Inject ice telemetry signatures
        cpr[ice_mask] = np.random.uniform(1.1, 1.6, size=np.sum(ice_mask))
        dop[ice_mask] = np.random.uniform(0.04, 0.12, size=np.sum(ice_mask))
        
        # Clip values to physically realistic limits
        cpr = np.clip(cpr, 0.01, 2.5)
        dop = np.clip(dop, 0.01, 1.0)
        
        return cpr, dop

    def _generate_rock_density(self, slope):
        """
        Computes rock density. Steep slopes trigger landslides/ejecta, yielding higher rock density.
        """
        base_rocks = np.random.poisson(lam=1.5, size=(self.size, self.size)) / 5.0
        slope_rocks = (slope / 20.0)**2 * np.random.uniform(0.2, 0.8, size=(self.size, self.size))
        rock_density = np.clip(base_rocks + slope_rocks, 0.0, 1.0)
        return rock_density

    def calculate_shadows(self, dem, sun_azimuth_deg, sun_elevation_deg):
        """
        Computes solar shadows across the terrain using a fast ray-tracing algorithm in NumPy.
        Returns:
            shadow_map: 2D matrix where 1 = illuminated, 0 = shadowed (PSR).
        """
        if sun_elevation_deg <= 0:
            return np.zeros_like(dem)  # Total darkness (lunar night)
            
        sun_azimuth_rad = np.radians(sun_azimuth_deg)
        sun_elevation_rad = np.radians(sun_elevation_deg)
        
        # Direction vector of incoming sunlight
        dx = -np.cos(sun_azimuth_rad)
        dy = -np.sin(sun_azimuth_rad)
        tan_elev = np.tan(sun_elevation_rad)
        
        shadow_map = np.ones_like(dem)
        size = self.size
        
        # We step along the sun vector to determine if any terrain point blocks the sun
        # Max horizontal ray length is bound by crater width
        max_steps = int(size * 0.8)
        step_dist = self.resolution
        
        # Move across the grid
        for step in range(1, max_steps):
            shift_x = int(round(step * dx))
            shift_y = int(round(step * dy))
            
            if abs(shift_x) >= size or abs(shift_y) >= size:
                break
                
            # Shifted DEM represents the height of potential blocker terrain
            # As the light moves from the blocker, it descends by step_dist * step * tan_elev
            shifted_dem = np.zeros_like(dem)
            
            # Safe boundary shifting
            slice_y_src = slice(max(0, -shift_y), min(size, size - shift_y))
            slice_y_dst = slice(max(0, shift_y), min(size, size + shift_y))
            slice_x_src = slice(max(0, -shift_x), min(size, size - shift_x))
            slice_x_dst = slice(max(0, shift_x), min(size, size + shift_x))
            
            shifted_dem[slice_y_dst, slice_x_dst] = dem[slice_y_src, slice_x_src]
            
            # Elevation threshold that light must clear to illuminate target point
            light_height = shifted_dem - (step * step_dist * tan_elev)
            
            # If light_height is higher than current dem, it means it's blocked
            blocked = light_height > dem
            shadow_map[blocked] = 0.0
            
        return shadow_map

    def inject_telemetry_glitches(self, matrix, glitch_rate=0.05, stripe_prob=0.02):
        """
        Injects simulated transmission dropouts:
        - Salt and Pepper (nan/zeros)
        - Horizontal stripe dropouts (telemetry sync losses)
        """
        corrupted = matrix.copy()
        h, w = corrupted.shape
        
        # 1. Point anomalies (hot pixels / dropped telemetry packets)
        num_glitches = int(glitch_rate * corrupted.size)
        y_indices = np.random.randint(0, h, num_glitches)
        x_indices = np.random.randint(0, w, num_glitches)
        
        # Some are zero, some are extreme noise spikes (e.g. 9.99 CPR), some are NaN
        glitch_types = np.random.choice(['zero', 'spike', 'nan'], size=num_glitches)
        for idx in range(num_glitches):
            g_type = glitch_types[idx]
            y, x = y_indices[idx], x_indices[idx]
            if g_type == 'zero':
                corrupted[y, x] = 0.0
            elif g_type == 'spike':
                corrupted[y, x] = 9.9  # Extreme invalid value
            else:
                corrupted[y, x] = np.nan
                
        # 2. Horizontal stripe dropouts (common in push-broom sensors)
        num_stripes = int(stripe_prob * h)
        stripe_rows = np.random.randint(0, h, num_stripes)
        for r in stripe_rows:
            corrupted[r, :] = 0.0  # Entire line loss
            
        return corrupted

# Graceful loading of files helper
def load_lunar_geotiff(filepath):
    """
    Loads raw geotiff if rasterio is available. Falls back to generating crater data structure.
    """
    if not RASTERIO_AVAILABLE:
        return None
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            # Simple conversion to float32
            data = data.astype(np.float32)
            # Remove bad values
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            return {
                "data": data,
                "bounds": src.bounds,
                "crs": src.crs.to_string() if src.crs else "Unknown",
                "transform": list(src.transform)
            }
    except Exception as e:
        print(f"Error loading GeoTIFF: {e}")
        return None

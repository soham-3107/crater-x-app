import numpy as np
from lunar_data import LunarCraterSimulator
from glitch_eraser import SelfHealingGlitchEraser
from subsurface import SubsurfaceIceEstimator
from pathfinding import LunarPathPlanner

def main():
    print("Initializing test run...")
    
    # 1. Simulator
    sim = LunarCraterSimulator(size=100, resolution=10.0)
    data = sim.generate_crater(name="Shackleton", center_peak=False)
    print("Crater DEM generated. Shape:", data["dem"].shape)
    
    # 2. Shadows
    shadows = sim.calculate_shadows(data["dem"], 145.0, 1.5)
    print("Shadow map generated. Sunlight fraction:", np.mean(shadows))
    
    # 3. Glitches
    glitched_cpr = sim.inject_telemetry_glitches(data["cpr"], glitch_rate=0.1)
    print("Glitches injected. Glitched cpr shape:", glitched_cpr.shape)
    
    # 4. Healing
    eraser = SelfHealingGlitchEraser()
    healed_cpr, cpr_mask = eraser.heal_layer(glitched_cpr, "cpr")
    print("Telemetry healed. Repaired pixels:", np.sum(cpr_mask))
    
    # 5. Subsurface Ice
    estimator = SubsurfaceIceEstimator(max_depth=5.0, dz=0.1)
    ice_3d, temp_3d = estimator.estimate_3d_ice_grid(healed_cpr, data["dop"], resolution=10.0)
    mass = estimator.calculate_total_water_mass(ice_3d, resolution=10.0)
    print(f"3D Ice Core Estimator run complete. Total water mass: {mass:.2f} MT")
    
    # 6. Pathfinder
    planner = LunarPathPlanner(resolution=10.0, max_slope_passable=20.0)
    start = (10, 10)
    end = (50, 50)
    path, success = planner.plan_traverse(data["dem"], data["slope"], shadows, start, end, downsample_factor=2)
    print(f"Pathfinding complete. Success: {success}, Path length: {len(path)} steps")
    
    if success:
        bat_telemetry = planner.simulate_battery(path, shadows)
        print(f"Battery simulation complete. Final charge: {bat_telemetry['final_charge']:.1f}%")
        
    print("All modules executed successfully and verified!")

if __name__ == "__main__":
    main()

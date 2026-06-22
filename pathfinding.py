import numpy as np
import networkx as nx

class LunarPathPlanner:
    def __init__(self, resolution=10.0, max_slope_passable=20.0, slope_penalty_threshold=12.0):
        """
        resolution: meters per pixel
        max_slope_passable: slope in degrees above which the rover cannot traverse
        slope_penalty_threshold: slope in degrees above which the traversability cost increases exponentially
        """
        self.resolution = resolution
        self.max_slope_passable = max_slope_passable
        self.slope_penalty_threshold = slope_penalty_threshold

    def plan_traverse(self, dem, slope, shadow_map, start, end, downsample_factor=2, shadow_penalty_factor=5.0):
        """
        Plans an optimal path from start (y, x) to end (y, x) using A* search.
        
        downsample_factor: Downsamples the grid to speed up graph computations (highly responsive UI).
        shadow_penalty_factor: Penalty multiplier for travelling through shadows.
        """
        h, w = dem.shape
        
        # 1. Downsample matrices
        ds_h, ds_w = h // downsample_factor, w // downsample_factor
        
        # Resize using basic slicing
        dem_ds = dem[::downsample_factor, ::downsample_factor]
        slope_ds = slope[::downsample_factor, ::downsample_factor]
        shadow_ds = shadow_map[::downsample_factor, ::downsample_factor]
        
        ds_res = self.resolution * downsample_factor
        
        # Map original coords to downsampled coords
        start_ds = (start[0] // downsample_factor, start[1] // downsample_factor)
        end_ds = (end[0] // downsample_factor, end[1] // downsample_factor)
        
        # Clamp coordinates to downsampled grid boundaries
        start_ds = (np.clip(start_ds[0], 0, ds_h - 1), np.clip(start_ds[1], 0, ds_w - 1))
        end_ds = (np.clip(end_ds[0], 0, ds_h - 1), np.clip(end_ds[1], 0, ds_w - 1))
        
        # 2. Build NetworkX Grid Graph
        G = nx.Graph()
        
        # Nodes
        for y in range(ds_h):
            for x in range(ds_w):
                G.add_node((y, x), elevation=dem_ds[y, x], slope=slope_ds[y, x], sun=shadow_ds[y, x])
                
        # Edges (8-way connectivity)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for y in range(ds_h):
            for x in range(ds_w):
                u = (y, x)
                s_u = slope_ds[y, x]
                
                # Skip if impassable
                if s_u > self.max_slope_passable:
                    continue
                    
                for dy, dx in directions:
                    ny, nx_val = y + dy, x + dx
                    
                    if 0 <= ny < ds_h and 0 <= nx_val < ds_w:
                        v = (ny, nx_val)
                        s_v = slope_ds[ny, nx_val]
                        
                        # Skip if destination is impassable
                        if s_v > self.max_slope_passable:
                            continue
                            
                        # Calculate distance cost
                        dist = np.sqrt(dy**2 + dx**2) * ds_res
                        
                        # Slope penalty (exponential factor for steep climbs)
                        avg_slope = (s_u + s_v) / 2.0
                        if avg_slope > self.slope_penalty_threshold:
                            slope_multiplier = 1.0 + 15.0 * ((avg_slope - self.slope_penalty_threshold) / (self.max_slope_passable - self.slope_penalty_threshold))**3
                        else:
                            slope_multiplier = 1.0
                            
                        # Shadow penalty (protect battery life in PSRs)
                        # Sun: 1.0 = illuminated, 0.0 = shadow
                        avg_sun = (shadow_ds[y, x] + shadow_ds[ny, nx_val]) / 2.0
                        sun_multiplier = 1.0 + (1.0 - avg_sun) * (shadow_penalty_factor - 1.0)
                        
                        # Combined edge weight
                        weight = dist * slope_multiplier * sun_multiplier
                        
                        G.add_edge(u, v, weight=weight)
                        
        # 3. Solve Path using A*
        def heuristic(node1, node2):
            # Straight-line Euclidean distance
            return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2) * ds_res
            
        try:
            path_ds = nx.astar_path(G, start_ds, end_ds, heuristic=heuristic, weight='weight')
            
            # Map back to original coordinate system
            path_orig = [(p[0] * downsample_factor, p[1] * downsample_factor) for p in path_ds]
            return path_orig, True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback path if graph is disconnected: straight line for rendering, flag failure
            return [], False

    def simulate_battery(self, path, shadow_map, start_charge=100.0, drain_rate=0.8, recharge_rate=1.2):
        """
        Simulates the battery charge of the rover as it traverses the path.
        
        path: List of (y, x) coordinates in the original grid.
        shadow_map: 2D binary grid (1 = sunlight, 0 = shadow).
        start_charge: initial charge (%).
        drain_rate: battery % loss per step in shadow.
        recharge_rate: battery % gained per step in sunlight.
        """
        if not path:
            return []
            
        battery_levels = [start_charge]
        current_charge = start_charge
        
        # Track timeline status
        shadow_duration = 0
        sun_duration = 0
        events = []
        
        for idx in range(1, len(path)):
            y, x = path[idx]
            is_sunlight = shadow_map[y, x] > 0.5
            
            if is_sunlight:
                current_charge += recharge_rate
                sun_duration += 1
                shadow_duration = 0
            else:
                current_charge -= drain_rate
                shadow_duration += 1
                sun_duration = 0
                
            current_charge = np.clip(current_charge, 0.0, 100.0)
            battery_levels.append(current_charge)
            
            # Warning checks
            if current_charge <= 10.0 and len(events) == 0 or (len(events) > 0 and events[-1]['type'] != 'CRITICAL' and current_charge < 5.0):
                events.append({
                    "step": idx,
                    "type": "CRITICAL",
                    "msg": f"Battery Critically Low ({current_charge:.1f}%) at step {idx}!"
                })
                
        return {
            "battery_profile": battery_levels,
            "min_charge": min(battery_levels),
            "final_charge": battery_levels[-1],
            "shadow_steps": sum([1 for p in path if shadow_map[p[0], p[1]] < 0.5]),
            "sun_steps": sum([1 for p in path if shadow_map[p[0], p[1]] >= 0.5]),
            "events": events
        }

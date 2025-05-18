import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import math
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
import streamlit as st
from collections import deque
import time

# ======================== SHAPE GENERATORS ========================
def circle_shape(r, cx=0, cy=0, segments=20):
    """Generate a circle with given radius and center"""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def polygon_shape(n_sides, r, cx=0, cy=0):
    """Generate a regular polygon with n sides"""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

def ellipse_shape(r, aspect_ratio, cx=0, cy=0, segments=30):
    """Generate an ellipse with given aspect ratio"""
    angles = np.linspace(0, 2 * np.pi, segments)
    return [(cx + r * np.cos(a), cy + r * aspect_ratio * np.sin(a)) for a in angles]

def star_shape(n_points, r, cx=0, cy=0):
    """Generate a star shape with n points"""
    angles = np.linspace(0, 2 * np.pi, n_points * 2, endpoint=False)
    radius = [r if i % 2 == 0 else r * 0.5 for i in range(len(angles))]
    return [(cx + radius[i] * np.cos(angles[i]), cy + radius[i] * np.sin(angles[i])) for i in range(len(angles))]

def superellipse_shape(r, n=4, cx=0, cy=0, segments=50):
    """Generate a superellipse (squircle) shape"""
    angles = np.linspace(0, 2 * np.pi, segments)
    points = []
    for a in angles:
        x = r * np.sign(np.cos(a)) * abs(np.cos(a))**(2/n)
        y = r * np.sign(np.sin(a)) * abs(np.sin(a))**(2/n)
        points.append((cx + x, cy + y))
    return points

def lemniscate_shape(r, cx=0, cy=0, segments=50):
    """Generate a lemniscate (infinity symbol) shape"""
    t = np.linspace(0, 2 * np.pi, segments)
    a = r / np.sqrt(2)
    x = a * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    y = a * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)
    return [(cx + x[i], cy + y[i]) for i in range(len(t))]

def gear_shape(r, teeth=12, pressure_angle=20, cx=0, cy=0):
    """Generate a non-overlapping gear shape with specified teeth"""
    # Corrected gear profile generation
    pa = math.radians(pressure_angle)
    pitch_radius = r
    base_radius = pitch_radius * math.cos(pa)
    addendum = 0.8 * (2 * math.pi * pitch_radius / teeth)  # Reduced addendum to prevent overlap
    dedendum = 1.25 * addendum
    
    outer_radius = pitch_radius + addendum
    root_radius = pitch_radius - dedendum
    root_radius = max(root_radius, base_radius * 0.9)  # Ensure root radius doesn't get too small
    
    points = []
    for i in range(teeth * 2):
        angle = 2 * math.pi * i / (teeth * 2)
        
        if i % 2 == 0:  # Tooth tip
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
        else:  # Tooth root
            x = root_radius * math.cos(angle)
            y = root_radius * math.sin(angle)
        
        points.append((cx + x, cy + y))
    
    return points

def hypotrochoid_shape(r, cx=0, cy=0, segments=100):
    """Generate a hypotrochoid shape"""
    t = np.linspace(0, 2 * np.pi, segments)
    a, b, h = r, r/3, r/2
    x = (a - b) * np.cos(t) + h * np.cos((a - b)/b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b)/b * t)
    return [(cx + x[i], cy + y[i]) for i in range(len(t))]

def flower_shape(r, petals=5, cx=0, cy=0, segments=50):
    """Generate a flower shape with specified petals"""
    angles = np.linspace(0, 2 * np.pi, segments)
    points = []
    for a in angles:
        radius = r * (0.5 + 0.5 * np.sin(petals * a))
        x = cx + radius * np.cos(a)
        y = cy + radius * np.sin(a)
        points.append((x, y))
    return points

def random_shape(r, cx=0, cy=0, segments=30):
    """Generate a random organic shape"""
    base_circle = circle_shape(r, cx, cy, segments)
    base_circle = np.array(base_circle)
    
    # Add random perturbations
    perturbations = np.random.normal(0, 0.2*r, (segments, 2))
    smoothed_perturbations = np.zeros_like(perturbations)
    
    # Smooth the perturbations
    for i in range(2):
        tck, _ = splprep([perturbations[:, i]], s=segments)
        smoothed_perturbations[:, i] = splev(np.linspace(0, 1, segments), tck)[0]
    
    random_shape = base_circle + smoothed_perturbations
    return [(p[0], p[1]) for p in random_shape]

# ======================== THREAD GENERATION ========================
def generate_threads(Lx, Ly, min_r, max_r, num_threads, materials, packing='grid'):
    """
    Generate thread positions with different packing algorithms
    packing options: 'grid', 'random', 'hexagonal', 'poisson'
    """
    threads = []
    placed = 0
    max_attempts = num_threads * 1000
    
    if packing == 'grid':
        # Grid-based placement with jitter
        grid_size = int(np.sqrt(num_threads) * 1.2)
        x_grid = np.linspace(max_r, Lx - max_r, grid_size)
        y_grid = np.linspace(max_r, Ly - max_r, grid_size)
        
        for x in x_grid:
            for y in y_grid:
                if placed >= num_threads:
                    break
                
                # Add some jitter to the grid positions
                jitter_x = np.random.uniform(-max_r/2, max_r/2)
                jitter_y = np.random.uniform(-max_r/2, max_r/2)
                x_jittered = x + jitter_x
                y_jittered = y + jitter_y
                
                if x_jittered < max_r or x_jittered > Lx - max_r:
                    continue
                if y_jittered < max_r or y_jittered > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x_jittered, y_jittered, r, threads):
                    threads.append((x_jittered, y_jittered, r, material, waviness, roughness, max_length))
                    placed += 1
    
    elif packing == 'hexagonal':
        # Hexagonal close packing
        spacing = max_r * 2 * 1.05  # 5% spacing between fibers
        rows = int((Ly - 2*max_r) / (spacing * np.sqrt(3)/2)) + 1
        cols = int((Lx - 2*max_r) / spacing) + 1
        
        for row in range(rows):
            for col in range(cols):
                if placed >= num_threads:
                    break
                
                x = max_r + col * spacing
                if row % 2 == 1:
                    x += spacing / 2
                
                y = max_r + row * spacing * np.sqrt(3)/2
                
                if x > Lx - max_r or y > Ly - max_r:
                    continue
                
                r = np.random.uniform(min_r, max_r)
                material = random.choice(materials)
                waviness = np.random.uniform(0.05 * r, 0.2 * r)
                roughness = np.random.uniform(0.03, 0.1)
                max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
                
                if not check_collision(x, y, r, threads):
                    threads.append((x, y, r, material, waviness, roughness, max_length))
                    placed += 1
    
    elif packing == 'poisson':
        # Optimized Poisson disk sampling using Bridson's algorithm
        cell_size = min_r / np.sqrt(2)
        grid_width = int(Lx / cell_size) + 1
        grid_height = int(Ly / cell_size) + 1
        
        # Initialize grid
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        process_list = deque()
        sample_points = []
        
        # First sample
        x, y = np.random.uniform(max_r, Lx - max_r), np.random.uniform(max_r, Ly - max_r)
        process_list.append((x, y))
        sample_points.append((x, y))
        grid[int(x/cell_size)][int(y/cell_size)] = (x, y)
        
        k = 30  # Number of attempts before rejection
        
        while process_list and placed < num_threads:
            idx = np.random.randint(0, len(process_list))
            x, y = process_list[idx]
            
            found = False
            for _ in range(k):
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.uniform(2*min_r, 2*max_r)
                new_x = x + distance * np.cos(angle)
                new_y = y + distance * np.sin(angle)
                
                if (new_x < max_r or new_x > Lx - max_r or 
                    new_y < max_r or new_y > Ly - max_r):
                    continue
                
                # Check neighboring cells
                grid_x, grid_y = int(new_x/cell_size), int(new_y/cell_size)
                valid = True
                
                for i in range(max(0, grid_x-2), min(grid_width, grid_x+3)):
                    for j in range(max(0, grid_y-2), min(grid_height, grid_y+3)):
                        if grid[i][j] is not None:
                            px, py = grid[i][j]
                            if np.sqrt((px - new_x)**2 + (py - new_y)**2) < 2*min_r:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    process_list.append((new_x, new_y))
                    sample_points.append((new_x, new_y))
                    grid[grid_x][grid_y] = (new_x, new_y)
                    found = True
                    break
            
            if not found:
                process_list.remove((x, y))
        
        for x, y in sample_points[:num_threads]:
            r = np.random.uniform(min_r, max_r)
            material = random.choice(materials)
            waviness = np.random.uniform(0.05 * r, 0.2 * r)
            roughness = np.random.uniform(0.03, 0.1)
            max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
            threads.append((x, y, r, material, waviness, roughness, max_length))
            placed += 1
    
    else:  # random
        # Pure random placement with collision checking
        attempts = 0
        while placed < num_threads and attempts < max_attempts:
            x = np.random.uniform(max_r, Lx - max_r)
            y = np.random.uniform(max_r, Ly - max_r)
            r = np.random.uniform(min_r, max_r)
            material = random.choice(materials)
            waviness = np.random.uniform(0.05 * r, 0.2 * r)
            roughness = np.random.uniform(0.03, 0.1)
            max_length = 1.0 if placed < num_threads * 0.2 else np.random.uniform(0.3, 0.9)
            
            if not check_collision(x, y, r, threads):
                threads.append((x, y, r, material, waviness, roughness, max_length))
                placed += 1
            attempts += 1
    
    placement_ratio = placed / num_threads * 100
    st.write(f"Successfully placed {placed}/{num_threads} threads ({placement_ratio:.1f}%) using {packing} packing")
    
    if placement_ratio < 90:
        st.warning("Low placement percentage. Consider increasing domain size or decreasing thread count.")
    
    return threads

def check_collision(x, y, r, existing_threads, safety_margin=1.1):
    """Check if a new thread would collide with existing ones"""
    if not existing_threads:
        return False
    
    existing_pos = np.array([[t[0], t[1]] for t in existing_threads])
    existing_rad = np.array([t[2] * safety_margin for t in existing_threads])
    
    distances = np.sqrt((existing_pos[:, 0] - x)**2 + (existing_pos[:, 1] - y)**2)
    return np.any(distances < (r * safety_margin + existing_rad))

# ======================== 3D THREAD RENDERING ========================
def generate_thread_layers(x, y, r, waviness, roughness, height, segments, shape_type, shape_params):
    """Generate 3D layers for a single thread"""
    layers = []
    z_values = np.linspace(0, height, segments + 1)
    
    # Generate base shape
    if shape_type == "circle":
        base_shape = circle_shape(r, 0, 0)
    elif shape_type == "polygon":
        base_shape = polygon_shape(shape_params['sides'], r, 0, 0)
    elif shape_type == "ellipse":
        base_shape = ellipse_shape(r, shape_params['aspect_ratio'], 0, 0)
    elif shape_type == "star":
        base_shape = star_shape(shape_params['points'], r, 0, 0)
    elif shape_type == "superellipse":
        base_shape = superellipse_shape(r, shape_params['n'], 0, 0)
    elif shape_type == "lemniscate":
        base_shape = lemniscate_shape(r, 0, 0)
    elif shape_type == "gear":
        base_shape = gear_shape(r, shape_params['teeth'], shape_params.get('pressure_angle', 20), 0, 0)
    elif shape_type == "hypotrochoid":
        base_shape = hypotrochoid_shape(r, 0, 0)
    elif shape_type == "flower":
        base_shape = flower_shape(r, shape_params['petals'], 0, 0)
    elif shape_type == "random":
        base_shape = random_shape(r, 0, 0)
    else:
        base_shape = circle_shape(r, 0, 0)
    
    base_shape = np.array(base_shape)
    
    # Add some twist to the shape
    twist_factor = shape_params.get('twist', 0)  # rotations per unit height
    
    for i, z in enumerate(z_values):
        # Current radius modulation
        r_mod = r * (1 + roughness * np.sin(2 * np.pi * z / height))
        
        # Position modulation (waviness)
        cx = x + waviness * np.sin(2 * np.pi * z / height)
        cy = y + waviness * np.cos(2 * np.pi * z / height)
        
        # Scale and rotate the base shape
        scaled_shape = base_shape * (r_mod / r)
        
        # Apply twist if specified
        if twist_factor != 0:
            angle = 2 * np.pi * z * twist_factor
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
            scaled_shape = np.dot(scaled_shape, rot_matrix)
        
        # Translate to final position
        translated_shape = scaled_shape + np.array([cx, cy])
        layers.append([(p[0], p[1], z) for p in translated_shape])
    
    return layers

def draw_threads(ax, threads, Lz, shape_type, shape_params, segments=20, color_by='material'):
    """Draw all threads in 3D space"""
    # Create colormaps for different coloring schemes
    if color_by == 'material':
        materials = list(set([t[3] for t in threads]))
        material_colors = plt.cm.tab20(np.linspace(0, 1, len(materials)))
        color_map = {mat: color for mat, color in zip(materials, material_colors)}
    elif color_by == 'radius':
        radii = [t[2] for t in threads]
        norm = plt.Normalize(min(radii), max(radii))
        cmap = plt.cm.viridis
    elif color_by == 'height':
        heights = [t[6] for t in threads]
        norm = plt.Normalize(min(heights), max(heights))
        cmap = plt.cm.plasma
    
    for idx, (x, y, r, material, waviness, roughness, max_length) in enumerate(threads):
        height = Lz * max_length
        layers = generate_thread_layers(x, y, r, waviness, roughness, height, segments, shape_type, shape_params)
        
        # Determine color based on coloring scheme
        if color_by == 'material':
            color = color_map[material]
        elif color_by == 'radius':
            color = cmap(norm(r))
        elif color_by == 'height':
            color = cmap(norm(max_length))
        
        for i in range(len(layers) - 1):
            verts = []
            n = len(layers[i])
            for j in range(n):
                k = (j + 1) % n
                quad = [layers[i][j], layers[i][k], layers[i+1][k], layers[i+1][j]]
                verts.append(quad)
            
            poly = Poly3DCollection(verts, alpha=0.8)
            poly.set_facecolor(color)
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect([Lx, Ly, Lz])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add colorbar if coloring by radius or height
    if color_by in ['radius', 'height']:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Radius' if color_by == 'radius' else 'Height fraction')

# ======================== 2D PROJECTION ========================
def plot_2d_projections(threads, Lx, Ly, shape_type, shape_params):
    """Create 2D projection plots of the threads"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Top view (XY plane)
    ax1.set_title(f"Top View (XY Plane) - {shape_type} Cross-sections")
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    ax1.set_aspect('equal')
    
    # Side view (XZ plane)
    ax2.set_title(f"Side View (XZ Plane) - Thread Heights")
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Lz)
    ax2.set_aspect('auto')
    
    for x, y, r, material, _, _, max_length in threads:
        # Generate cross-section shape
        if shape_type == "circle":
            profile = circle_shape(r, 0, 0)
        elif shape_type == "polygon":
            profile = polygon_shape(shape_params['sides'], r, 0, 0)
        elif shape_type == "ellipse":
            profile = ellipse_shape(r, shape_params['aspect_ratio'], 0, 0)
        elif shape_type == "star":
            profile = star_shape(shape_params['points'], r, 0, 0)
        elif shape_type == "superellipse":
            profile = superellipse_shape(r, shape_params['n'], 0, 0)
        elif shape_type == "lemniscate":
            profile = lemniscate_shape(r, 0, 0)
        elif shape_type == "gear":
            profile = gear_shape(r, shape_params['teeth'], shape_params.get('pressure_angle', 20), 0, 0)
        elif shape_type == "hypotrochoid":
            profile = hypotrochoid_shape(r, 0, 0)
        elif shape_type == "flower":
            profile = flower_shape(r, shape_params['petals'], 0, 0)
        elif shape_type == "random":
            profile = random_shape(r, 0, 0)
        
        # Top view projection
        translated = [(x + p[0], y + p[1]) for p in profile]
        poly = plt.Polygon(translated, alpha=0.6, edgecolor='k')
        ax1.add_patch(poly)
        
        # Side view projection
        height = Lz * max_length
        ax2.plot([x - r, x + r], [height, height], 'b-', alpha=0.5)
        ax2.plot([x, x], [0, height], 'r-', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# ======================== STATISTICAL ANALYSIS ========================
def analyze_thread_distribution(threads):
    """Calculate and display statistics about the thread distribution"""
    radii = [t[2] for t in threads]
    heights = [t[6] for t in threads]
    materials = [t[3] for t in threads]
    
    st.write("\n### Thread Distribution Analysis")
    st.write(f"Total threads: {len(threads)}")
    st.write(f"Radius stats - Min: {min(radii):.3f}, Max: {max(radii):.3f}, Mean: {np.mean(radii):.3f}")
    st.write(f"Height stats - Min: {min(heights):.3f}, Max: {max(heights):.3f}, Mean: {np.mean(heights):.3f}")
    
    # Material distribution
    material_counts = {mat: materials.count(mat) for mat in set(materials)}
    st.write("\n**Material distribution:**")
    for mat, count in material_counts.items():
        st.write(f"- {mat}: {count} threads ({count/len(threads)*100:.1f}%)")
    
    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(radii, bins=20, edgecolor='black')
    ax1.set_title('Thread Radius Distribution')
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Count')
    
    ax2.hist(heights, bins=20, edgecolor='black')
    ax2.set_title('Thread Height Distribution')
    ax2.set_xlabel('Height Fraction')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    st.pyplot(fig)

# ======================== STREAMLIT APP ========================
def main():
    global Lx, Ly, Lz
    
    st.title("Advanced Composite Thread Simulation")
    st.write("This app simulates composite materials with different thread shapes and packing algorithms.")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    shape_options = ["circle", "polygon", "ellipse", "star", "superellipse", 
                    "lemniscate", "gear", "hypotrochoid", "flower", "random"]
    shape_type = st.sidebar.selectbox("Thread shape", shape_options, index=0)
    
    shape_params = {}
    if shape_type == "polygon":
        shape_params['sides'] = st.sidebar.slider("Number of sides", 3, 12, 6)
    elif shape_type == "ellipse":
        shape_params['aspect_ratio'] = st.sidebar.slider("Aspect ratio", 0.1, 5.0, 1.5)
    elif shape_type == "star":
        shape_params['points'] = st.sidebar.slider("Number of points", 5, 20, 5)
    elif shape_type == "superellipse":
        shape_params['n'] = st.sidebar.slider("Exponent (2=ellipse, 4=rounded square)", 2.0, 10.0, 4.0)
    elif shape_type == "gear":
        shape_params['teeth'] = st.sidebar.slider("Number of teeth", 8, 20, 12)
        shape_params['pressure_angle'] = st.sidebar.slider("Pressure angle (degrees)", 10, 30, 20)
    elif shape_type == "flower":
        shape_params['petals'] = st.sidebar.slider("Number of petals", 3, 12, 5)
    elif shape_type == "random":
        shape_params['twist'] = st.sidebar.slider("Twist factor", 0.0, 2.0, 0.5)
    
    Lx = st.sidebar.slider("Domain length (X)", 1.0, 20.0, 10.0)
    Ly = st.sidebar.slider("Domain width (Y)", 1.0, 20.0, 10.0)
    Lz = st.sidebar.slider("Domain height (Z)", 1.0, 20.0, 5.0)
    num_threads = st.sidebar.slider("Number of threads", 1, 1000, 100)
    min_d = st.sidebar.slider("Minimum diameter", 0.05, 1.0, 0.2)
    max_d = st.sidebar.slider("Maximum diameter", 0.05, 1.0, 0.4)
    
    packing_options = ['grid', 'hexagonal', 'poisson', 'random']
    packing = st.sidebar.selectbox("Packing algorithm", packing_options, index=1)
    
    materials = ["carbon", "glass", "kevlar", "steel", "nylon", "copper"]
    selected_materials = st.sidebar.multiselect("Materials", materials, default=["carbon", "glass", "kevlar"])
    
    if not selected_materials:
        st.sidebar.warning("Please select at least one material")
        return
    
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Generating threads..."):
            start_time = time.time()
            threads = generate_threads(Lx, Ly, min_d/2, max_d/2, num_threads, selected_materials, packing)
            st.success(f"Thread generation completed in {time.time() - start_time:.2f} seconds")
        
        # Statistical analysis
        analyze_thread_distribution(threads)
        
        # 3D Visualization
        with st.spinner("Generating 3D visualization..."):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            draw_threads(ax, threads, Lz, shape_type, shape_params, color_by='material')
            ax.set_title(f"{len(threads)} {shape_type} threads ({packing} packing)")
            plt.tight_layout()
            st.pyplot(fig)
        
        # 2D Projections
        with st.spinner("Generating 2D projections..."):
            plot_2d_projections(threads, Lx, Ly, shape_type, shape_params)

if __name__ == "__main__":
    main()

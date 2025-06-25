# -*- coding: utf-8 -*-
"""
2D Cosmological N-Body Simulation (Revised Version)

This script has been revised to precisely match the initial conditions,
simulation parameters, and visualization style of the user-provided
reference code (nbody.py, cft.py, phase_plot.py).

Key changes:
1.  **Initial Conditions**: Replicated the specific power spectrum
    (Power_law * Scale * Cutoff), random seed, and amplitude from the
    reference code to generate an identical starting state.
2.  **Simulation Run**: Extended the simulation end time and now saves a
    snapshot at every time step to match the original behavior.
3.  **Visualization**: The plotting function now uses the same color map,
    alpha, value ranges, and area normalization. The final output is a
    2x3 collage showing full and zoomed views, identical to phase_plot.py.
"""
import os
import numpy as np
from numpy import fft
from numba import njit
from matplotlib import pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass

# --- Matplotlib Appearance Setup ---
rcParams["font.family"] = "serif"
rcParams["figure.dpi"] = 150

# --- Simulation & Cosmological Parameters ---
N_PARTICLES_PER_DIM = 256
BOX_SIZE = 50.0
GRID_SIZE = 512

# Time evolution parameters adjusted to match reference
SCALE_FACTOR_BEGIN = 0.02
SCALE_FACTOR_END = 2.1  # Run longer to get snapshots at a=2.0
TIME_STEP_SIZE = 0.02
N_TIME_STEPS = int((SCALE_FACTOR_END - SCALE_FACTOR_BEGIN) / TIME_STEP_SIZE)

@dataclass
class Cosmology:
    """A simple class to hold cosmological parameters for EdS universe."""
    OmegaM: float = 1.0
    
    @property
    def G_eff(self):
        """Effective gravitational constant for the simulation."""
        return 1.5 * self.OmegaM

COSMOLOGY = Cosmology()

# --- Utility Functions & Classes ---
def get_snapshot_path(scale_factor):
    """Generates the file path for a given snapshot, matching original format."""
    if not os.path.exists('data'):
        os.makedirs('data')
    return f"data/x.{int(round(scale_factor*1000)):05d}.npy"

@dataclass
class Box:
    """A helper class to hold grid properties."""
    dim: int
    res: int
    size: float

    @property
    def shape(self):
        return tuple([self.res] * self.dim)
    
    @property
    def k_min(self):
        return 2 * np.pi / self.size
    
    @property
    def k_max(self):
        return np.pi * self.res / self.size

# =============================================================================
# PART 1: N-BODY SIMULATION CORE (ACCELERATED WITH NUMBA)
# =============================================================================

@njit
def deposit_mass_cic(positions, grid_shape, box_size):
    n_particles = positions.shape[0]
    grid_res = grid_shape[0]
    cell_size = box_size / grid_res
    density_grid = np.zeros(grid_shape, dtype=np.float64)
    
    pos_grid = positions / cell_size

    for i in range(n_particles):
        ix, iy = int(pos_grid[i, 0]), int(pos_grid[i, 1])
        fx, fy = pos_grid[i, 0] - ix, pos_grid[i, 1] - iy
        
        wx0, wx1 = (1 - fx), fx
        wy0, wy1 = (1 - fy), fy

        i_plus_1 = (ix + 1) % grid_res
        j_plus_1 = (iy + 1) % grid_res
        
        density_grid[ix, iy]           += wx0 * wy0
        density_grid[i_plus_1, iy]     += wx1 * wy0
        density_grid[ix, j_plus_1]     += wx0 * wy1
        density_grid[i_plus_1, j_plus_1] += wx1 * wy1

    mean_density = n_particles / (grid_res * grid_res)
    density_contrast = (density_grid / mean_density) - 1.0
    return density_contrast

@njit
def interpolate_force_cic(force_x_grid, force_y_grid, positions, box_size):
    n_particles = positions.shape[0]
    grid_res = force_x_grid.shape[0]
    cell_size = box_size / grid_res
    forces = np.empty_like(positions)

    pos_grid = positions / cell_size
    
    for i in range(n_particles):
        ix, iy = int(pos_grid[i, 0]), int(pos_grid[i, 1])
        fx, fy = pos_grid[i, 0] - ix, pos_grid[i, 1] - iy

        wx0, wx1 = (1 - fx), fx
        wy0, wy1 = (1 - fy), fy

        i_plus_1 = (ix + 1) % grid_res
        j_plus_1 = (iy + 1) % grid_res
        
        forces[i, 0] = (force_x_grid[ix, iy] * wx0 * wy0 +
                        force_x_grid[i_plus_1, iy] * wx1 * wy0 +
                        force_x_grid[ix, j_plus_1] * wx0 * wy1 +
                        force_x_grid[i_plus_1, j_plus_1] * wx1 * wy1)
        
        forces[i, 1] = (force_y_grid[ix, iy] * wx0 * wy0 +
                        force_y_grid[i_plus_1, iy] * wx1 * wy0 +
                        force_y_grid[ix, j_plus_1] * wx0 * wy1 +
                        force_y_grid[i_plus_1, j_plus_1] * wx1 * wy1)
    return forces

# =============================================================================
# PART 2: SIMULATION SETUP AND EXECUTION
# =============================================================================

def calculate_forces_pm(positions):
    grid_shape = (GRID_SIZE, GRID_SIZE)
    density_contrast = deposit_mass_cic(positions, grid_shape, BOX_SIZE)
    density_contrast_k = fft.fft2(density_contrast)
    
    k_freq = fft.fftfreq(GRID_SIZE, d=BOX_SIZE/GRID_SIZE)
    kx, ky = np.meshgrid(k_freq, k_freq, indexing='ij')
    k_squared = kx**2 + ky**2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        potential_k = -density_contrast_k / k_squared
    potential_k[0, 0] = 0
    
    force_k_x = -1j * kx * potential_k
    force_k_y = -1j * ky * potential_k

    force_grid_x = fft.ifft2(force_k_x).real
    force_grid_y = fft.ifft2(force_k_y).real

    return interpolate_force_cic(force_grid_x, force_grid_y, positions, BOX_SIZE)

def generate_initial_potential(box, seed, amplitude):
    """
    Generates the initial potential field, replicating the logic from
    the user's cft.py and nbody.py.
    """
    np.random.seed(seed)
    
    k_freq = fft.fftfreq(box.res, d=box.size/box.res)
    kx, ky = np.meshgrid(k_freq, k_freq, indexing='ij')
    k_squared = kx**2 + ky**2
    
    # --- Replicate cft.py filter logic ---
    # Power_law(-0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_abs = np.sqrt(k_squared)
        pk_filter = np.where(k_abs == 0, 0, k_abs**-0.5)

    # Scale(0.2)
    t = 0.2**2
    scale_filter = np.exp(t / (box.size/box.res)**2 * (np.cos(kx * (box.size/box.res)) - 1)) * \
                   np.exp(t / (box.size/box.res)**2 * (np.cos(ky * (box.size/box.res)) - 1))

    # Cutoff()
    cutoff_filter = np.where(k_squared <= box.k_max**2, 1, 0)
    
    # Combine filters
    total_power_spectrum = pk_filter * scale_filter * cutoff_filter

    # Generate random field with this power spectrum
    random_fourier = np.random.normal(0, 1, box.shape) + 1j * np.random.normal(0, 1, box.shape)
    phi_k = fft.fftn(random_fourier) * np.sqrt(total_power_spectrum)

    # Transform to potential (divide by -k^2)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_k = phi_k * np.where(k_squared==0, 0, -1./k_squared)
    phi_k[0,0] = 0.0

    # Return potential in real space, multiplied by amplitude
    return fft.ifftn(phi_k).real * amplitude


def initial_conditions():
    """Generates ICs using Zel'dovich approx. from the specified potential."""
    particle_box = Box(2, N_PARTICLES_PER_DIM, BOX_SIZE)
    
    # 1. Generate potential field matching the reference code
    phi = generate_initial_potential(particle_box, seed=4, amplitude=10)

    # 2. Get displacements from potential gradient
    phi_k = fft.fftn(phi)
    k_freq = fft.fftfreq(particle_box.res, d=particle_box.size/particle_box.res)
    kx, ky = np.meshgrid(k_freq, k_freq, indexing='ij')
    
    disp_k_x = -1j * kx * phi_k
    disp_k_y = -1j * ky * phi_k
    
    disp_x = fft.ifftn(disp_k_x).real
    disp_y = fft.ifftn(disp_k_y).real

    # 3. Create grid and apply displacements
    grid_coords = np.arange(0, BOX_SIZE, BOX_SIZE/N_PARTICLES_PER_DIM)
    x, y = np.meshgrid(grid_coords, grid_coords)
    
    positions = np.vstack([(x + SCALE_FACTOR_BEGIN * disp_x).ravel(), 
                           (y + SCALE_FACTOR_BEGIN * disp_y).ravel()]).T
    
    # 4. Set initial momenta
    momenta = np.vstack([(SCALE_FACTOR_BEGIN * disp_x).ravel(), 
                         (SCALE_FACTOR_BEGIN * disp_y).ravel()]).T
    
    positions %= BOX_SIZE
    return positions, momenta

def run_simulation():
    """Main simulation loop."""
    print("Starting 2D N-body simulation (Revised)...")
    positions, momenta = initial_conditions()
    
    scale_factors = np.arange(SCALE_FACTOR_BEGIN, SCALE_FACTOR_END, TIME_STEP_SIZE)

    print("Running time integration...")
    forces = calculate_forces_pm(positions) * COSMOLOGY.G_eff
    
    # Save initial state at a=0.02
    with open(get_snapshot_path(SCALE_FACTOR_BEGIN), "wb") as f:
            np.save(f, positions)
            np.save(f, momenta)

    for i in range(len(scale_factors) - 1):
        a = scale_factors[i]
        a_half = a + TIME_STEP_SIZE / 2.0
        
        momenta += forces * TIME_STEP_SIZE / a
        positions += (momenta / (a_half**2)) * TIME_STEP_SIZE
        positions %= BOX_SIZE
        
        forces = calculate_forces_pm(positions) * COSMOLOGY.G_eff
        
        a_next = scale_factors[i+1]
        print(f"  Step {i+1}/{len(scale_factors)-1}, Scale Factor a = {a_next:.3f}", end='\r')
        
        # Save snapshot at every step, as in the original code
        with open(get_snapshot_path(a_next), "wb") as f:
            np.save(f, positions)
            np.save(f, momenta)
            
    print("\nSimulation finished.")

# =============================================================================
# PART 3: VISUALIZATION (Matching phase_plot.py)
# =============================================================================

def box_triangles(box):
    idx = np.arange(box.res * box.res, dtype=int).reshape(box.shape)
    x0, x1 = idx[:-1, :-1], idx[:-1, 1:]
    x2, x3 = idx[1:, :-1], idx[1:, 1:]
    upper = np.array([x0, x1, x2]).transpose([1, 2, 0]).reshape([-1, 3])
    lower = np.array([x3, x2, x1]).transpose([1, 2, 0]).reshape([-1, 3])
    return np.r_[upper, lower]

def triangle_area(x, y, triangles):
    return 0.5 * (x[triangles[:,0]] * (y[triangles[:,1]] - y[triangles[:,2]]) +
                  x[triangles[:,1]] * (y[triangles[:,2]] - y[triangles[:,0]]) +
                  x[triangles[:,2]] * (y[triangles[:,0]] - y[triangles[:,1]]))

def plot_for_time(box, triangles, time, bbox, fig, ax):
    """Loads snapshot data and creates a density plot, matching phase_plot.py."""
    fn = get_snapshot_path(time)
    try:
        with open(fn, "rb") as f:
            positions = np.load(f)
            momenta = np.load(f)
    except FileNotFoundError:
        print(f"Warning: Snapshot file not found: {fn}")
        ax.text(0.5, 0.5, f'Data for a={time} not found', ha='center', va='center', color='red')
        ax.set_facecolor('black')
        return

    # Normalize area by cell size, as in the reference code
    area = abs(triangle_area(positions[:,0], positions[:,1], triangles)) / box.res**2
    area[area == 0] = 1e-20

    sorting = np.argsort(area)[::-1]
    
    # Plot log(1/area) which is a proxy for log density
    log_inv_area = np.log(1. / area)

    ax.tripcolor(positions[:,0], positions[:,1], triangles[sorting], log_inv_area[sorting],
                 alpha=0.3, vmin=-2, vmax=2, cmap='YlGnBu')
    
    ax.set_facecolor('black')
    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def create_plots():
    """Creates the final 2x3 collage of plots, matching phase_plot.py."""
    print("Generating plots...")
    particle_box = Box(2, N_PARTICLES_PER_DIM, BOX_SIZE)
    triangles = box_triangles(particle_box)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), facecolor='black')
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    plot_times = [0.5, 1.0, 2.0]
    
    # Top row: Full view
    for i, t in enumerate(plot_times):
        plot_for_time(particle_box, triangles, t, bbox=[(0,50), (0,50)], fig=fig, ax=axs[0,i])
        axs[0,i].set_title(f"a = {t}", color='white')

    # Bottom row: Zoomed view
    for i, t in enumerate(plot_times):
        plot_for_time(particle_box, triangles, t, bbox=[(15,30), (5, 20)], fig=fig, ax=axs[1,i])

    output_filename = 'cosmological_structure_matched.png'
    fig.savefig(output_filename, dpi=150, facecolor='black')
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    run_simulation()
    create_plots()


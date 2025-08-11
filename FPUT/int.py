import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Microsoft JhengHei')

# Coefficients for the 8th order symplectic integrator by Laskar & Robutel (2001).
# This is a symmetric 8-stage integrator of the form:
# S(dt) = product_{i=1 to 8} [exp(d_i * dt * A) * exp(c_i * dt * B)]
# where A is the drift operator (related to kinetic energy T) and B is the kick 
# operator (related to potential energy U).
# A corresponds to q_new = q + dt * (p/m)
# B corresponds to p_new = p + dt * F(q)
C_COEFFS = np.array([
    0.195557812560339,
    0.433890397482848,
    -0.207886431443621,
    0.078438221400434,
    0.078438221400434,
    -0.207886431443621,
    0.433890397482848,
    0.195557812560339,
])

D_COEFFS = np.array([
    0.0977789062801695,
    0.289196093121589,
    0.252813583900000,
    -0.139788583301759,
    -0.139788583301759,
    0.252813583900000,
    0.289196093121589,
    0.0977789062801695,
])

# A quick check to ensure the coefficients sum to 1.
assert np.isclose(np.sum(C_COEFFS), 1.0)
assert np.isclose(np.sum(D_COEFFS), 1.0)


@numba.njit
def _integration_loop(force_function, q_initial, p_initial, mass, dt, n_steps):
    """
    The core integration loop, accelerated with Numba.
    This function should not be called directly.
    """
    # Get the dimension of the system from the initial condition.
    dim = q_initial.shape[0]
    
    # Create arrays to store the full trajectory.
    q_traj = np.empty((n_steps + 1, dim))
    p_traj = np.empty((n_steps + 1, dim))
    
    # Set initial conditions.
    q_traj[0] = q_initial
    p_traj[0] = p_initial
    
    # Make copies of the current state to evolve in the loop.
    q_current = q_initial.copy()
    p_current = p_initial.copy()

    # Main integration loop.
    for i in range(n_steps):
        # A single time step `dt` is composed of 8 stages.
        for j in range(8):
            # Stage j: Drift step followed by a Kick step.
            q_current += D_COEFFS[j] * (p_current / mass) * dt
            p_current += C_COEFFS[j] * force_function(q_current) * dt
        
        # Store the results for this time step.
        q_traj[i + 1] = q_current
        p_traj[i + 1] = p_current
        
    return q_traj, p_traj


def symplectic_integrator_8th(force_function, y0, t_span, dt, mass):
    """
    A general-purpose 8th-order symplectic integrator.

    This function integrates a Hamiltonian system of the form H(q, p) = T(p) + U(q),
    where T(p) = p^2 / (2m) is the kinetic energy and U(q) is the potential energy.

    Args:
        force_function (callable): 
            A Numba-jitted function `f(q)` that computes the force 
            (i.e., -dU/dq) given the position vector `q`.
        y0 (tuple): 
            A tuple `(q0, p0)` containing the initial position and momentum vectors.
            `q0` and `p0` should be NumPy arrays.
        t_span (tuple): 
            A tuple `(t_start, t_end)` defining the time interval for integration.
        dt (float): 
            The fixed time step for the integration.
        mass (float): 
            The mass of the particle(s). Assumed to be a scalar.

    Returns:
        tuple: A tuple `(t, q_traj, p_traj)` where:
               - `t` is a NumPy array of time points.
               - `q_traj` is a NumPy array of position vectors at each time point.
               - `p_traj` is a NumPy array of momentum vectors at each time point.
    """
    t_start, t_end = t_span
    q0, p0 = y0

    # Ensure initial conditions are NumPy arrays for consistency.
    q0 = np.asarray(q0, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)

    if q0.ndim != 1 or p0.ndim != 1 or q0.shape != p0.shape:
        raise ValueError("q0 and p0 must be 1D NumPy arrays of the same shape.")

    if not isinstance(mass, (int, float)) or mass <= 0:
        raise ValueError("mass must be a positive number.")

    # Calculate the number of integration steps.
    n_steps = int(np.round((t_end - t_start) / dt))
    if n_steps <= 0:
        raise ValueError("t_end must be greater than t_start, resulting in a positive number of steps.")

    # Generate the array of time points for the output.
    t = np.linspace(t_start, t_end, n_steps + 1)

    # Run the Numba-accelerated integration loop.
    # The first call to a Numba function triggers compilation, which may take a moment.
    # Subsequent calls will be much faster.
    q_traj, p_traj = _integration_loop(force_function, q0, p0, mass, dt, n_steps)
    
    return t, q_traj, p_traj


# --- Example Usage: Simple Harmonic Oscillator (SHO) ---

@numba.njit
def harmonic_oscillator_force(q):
    """
    Force function for a simple harmonic oscillator.
    The Hamiltonian is H = p^2/(2m) + k*q^2/2.
    The force is F = -dU/dq = -k*q.
    For simplicity, we assume the spring constant k=1 here.
    """
    # spring constant k = 1.0
    return -1.0 * q

def run_sho_example():
    """
    Runs a simulation of a simple harmonic oscillator and plots the results
    to demonstrate the integrator's usage and verify its accuracy.
    """
    # --- System Parameters ---
    mass = 1.0
    k = 1.0  # The spring constant is defined in the force function.
    omega = np.sqrt(k / mass) # Angular frequency

    # --- Initial Conditions ---
    q0 = np.array([1.0]) # Initial position
    p0 = np.array([0.0]) # Initial momentum
    y0 = (q0, p0)

    # --- Integration Parameters ---
    t_span = (0, 100) # Integrate from t=0 to t=100
    dt = 0.1         # Time step

    # --- Run the Integrator ---
    print("Running 8th-order symplectic integrator for the SHO example...")
    t, q_traj, p_traj = symplectic_integrator_8th(
        force_function=harmonic_oscillator_force,
        y0=y0,
        t_span=t_span,
        dt=dt,
        mass=mass
    )
    print("Integration complete.")
    # Fix for the minus sign ('-') displaying as a box.
    # --- Analysis & Plotting ---
    
    # 1. Calculate the analytical solution for comparison.
    q_analytical = q0[0] * np.cos(omega * t)
    p_analytical = -mass * omega * q0[0] * np.sin(omega * t)
    
    # 2. Calculate the total energy of the numerical solution to check for conservation.
    # H = p^2/(2m) + k*q^2/2
    kinetic_energy = p_traj**2 / (2 * mass)
    potential_energy = k * q_traj**2 / 2
    total_energy = (kinetic_energy + potential_energy).flatten()
    initial_energy = total_energy[0]
    energy_error = (total_energy - initial_energy) / initial_energy

    # 3. Create plots to visualize the results.
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("八阶辛积分器：简谐振子示例 (8th Order Symplectic Integrator: SHO Example)", fontsize=16)

    # Plot 1: Phase Space (q vs p)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(q_traj, p_traj, label='数值解 (Numerical)', lw=2)
    ax1.plot(q_analytical, p_analytical, 'r--', label='解析解 (Analytical)', linewidth=1.5)
    ax1.set_title("相空间图 (Phase Space)")
    ax1.set_xlabel("位置 q (Position)")
    ax1.set_ylabel("动量 p (Momentum)")
    ax1.legend()
    ax1.axis('equal')

    # Plot 2: Position vs. Time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, q_traj, label='数值解 q(t) (Numerical)')
    ax2.plot(t, q_analytical, 'r--', label='解析解 q(t) (Analytical)', linewidth=1)
    ax2.set_title("位置 vs. 时间 (Position vs. Time)")
    ax2.set_xlabel("时间 t (Time)")
    ax2.set_ylabel("位置 q (Position)")
    ax2.legend()
    
    # Plot 3: Momentum vs. Time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, p_traj, label='数值解 p(t) (Numerical)')
    ax3.plot(t, p_analytical, 'r--', label='解析解 p(t) (Analytical)', linewidth=1)
    ax3.set_title("动量 vs. 时间 (Momentum vs. Time)")
    ax3.set_xlabel("时间 t (Time)")
    ax3.set_ylabel("动量 p (Momentum)")
    ax3.legend()

    # Plot 4: Relative Energy Error vs. Time
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(t, energy_error)
    ax4.set_title("相对能量误差 (Relative Energy Error)")
    ax4.set_xlabel("时间 t (Time)")
    ax4.set_ylabel("(E(t) - E(0)) / E(0)")
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    run_sho_example()
    from matplotlib.font_manager import fontManager
    for font in sorted(fontManager.get_font_names()):
        print(font)

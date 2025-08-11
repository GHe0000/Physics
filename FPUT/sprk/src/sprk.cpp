#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::Map;

// Coefficients (same as Python)
static const double C_COEFFS[8] = {
    0.195557812560339,
    0.433890397482848,
    -0.207886431443621,
    0.078438221400434,
    0.078438221400434,
    -0.207886431443621,
    0.433890397482848,
    0.195557812560339,
};

static const double D_COEFFS[8] = {
    0.0977789062801695,
    0.289196093121589,
    0.252813583900000,
    -0.139788583301759,
    -0.139788583301759,
    0.252813583900000,
    0.289196093121589,
    0.0977789062801695,
};

// Type for a C++ force functor: given q return f(q)
using ForceFuncCpp = std::function<VectorXd(const VectorXd&)>;

/*
  Core SPRK loop implemented in C++ using Eigen.
  Parameters:
    force_cpp: optional C++ functor. If force_py is provided (non-null), we'll call Python.
    force_py: optional py::function callback; if provided it will be used (slower).
    mass_vec: Eigen vector (size n) or scalar (len 1)
    q0_vec, p0_vec: initial conditions (size n)
    dt: timestep
    n_step: number of steps
  Returns tuple of numpy arrays (t, q_save, p_save)
*/
static py::tuple sprk_core(const py::object& force_obj,
                           const Eigen::VectorXd& mass_vec,
                           const Eigen::VectorXd& q0_vec,
                           const Eigen::VectorXd& p0_vec,
                           double dt,
                           int n_step,
                           bool force_is_python)
{
    int n = (int)q0_vec.size();

    // Prepare outputs
    // time array
    py::array_t<double> t_arr(n_step + 1);
    auto t_ptr = static_cast<double*>(t_arr.request().ptr);
    for (int i = 0; i <= n_step; ++i) t_ptr[i] = dt * i;

    // q_save (n_step+1, n)
    py::array_t<double> q_save_arr({n_step + 1, n});
    py::array_t<double> p_save_arr({n_step + 1, n});
    auto q_save_ptr = static_cast<double*>(q_save_arr.request().ptr);
    auto p_save_ptr = static_cast<double*>(p_save_arr.request().ptr);

    VectorXd q = q0_vec;
    VectorXd p = p0_vec;

    // Fill initial
    for (int j = 0; j < n; ++j) {
        q_save_ptr[j] = q(j);
        p_save_ptr[j] = p(j);
    }

    // Helper to compute acceleration/force
    std::function<VectorXd(const VectorXd&)> call_force;

    if (force_is_python) {
        // Wrap Python callable (py::object) into a C++ callable
        py::function f = py::reinterpret_borrow<py::function>(force_obj);
        call_force = [f](const VectorXd& q_local)->VectorXd {
            py::gil_scoped_acquire gil;
            // convert Eigen VectorXd -> numpy array
            py::array_t<double> q_py(q_local.size());
            auto qpy_ptr = static_cast<double*>(q_py.request().ptr);
            for (ssize_t i = 0; i < q_local.size(); ++i) qpy_ptr[i] = q_local[i];

            // call python function
            py::object res = f(q_py);
            // convert result back to VectorXd
            py::array_t<double> res_arr = py::cast<py::array_t<double>>(res);
            auto rptr = static_cast<double*>(res_arr.request().ptr);
            int m = (int)res_arr.request().shape[0];
            VectorXd out(m);
            for (int i = 0; i < m; ++i) out(i) = rptr[i];
            return out;
        };
    } else {
        // force_obj expected to be a capsule or omitted; will error if not provided
        throw std::runtime_error("C++ force functor mode not implemented in this wrapper. "
                                 "Use sprk_with_cpp_force when providing C++ functor.");
    }

    // Main loop
    for (int step = 0; step < n_step; ++step) {
        for (int j = 0; j < 8; ++j) {
            // q += D[j] * (p / mass) * dt
            for (int k = 0; k < n; ++k) {
                double m = (mass_vec.size() == 1) ? mass_vec[0] : mass_vec[k];
                q(k) += D_COEFFS[j] * (p(k) / m) * dt;
            }
            // p += C[j] * force(q) * dt
            VectorXd f = call_force(q);
            for (int k = 0; k < n; ++k) {
                p(k) += C_COEFFS[j] * f(k) * dt;
            }
        }
        // save
        for (int k = 0; k < n; ++k) {
            q_save_ptr[(step+1)*(size_t)n + k] = q(k);
            p_save_ptr[(step+1)*(size_t)n + k] = p(k);
        }
    }

    return py::make_tuple(t_arr, q_save_arr, p_save_arr);
}

/*
  A thin wrapper exposed to Python: expects a Python callable force_func
*/
py::tuple sprk_with_pyforce(py::function force_func,
                            py::object mass,
                            py::array_t<double> q0,
                            py::array_t<double> p0,
                            double t_total,
                            double dt)
{
    // convert initial arrays
    auto q0_r = q0.unchecked<1>();
    auto p0_r = p0.unchecked<1>();
    int n = (int)q0_r.shape(0);
    if ((int)p0_r.shape(0) != n) throw std::runtime_error("q0 and p0 must have same length");

    VectorXd q0_vec(n), p0_vec(n);
    for (int i = 0; i < n; ++i) { q0_vec[i] = q0_r(i); p0_vec[i] = p0_r(i); }

    // mass: can be scalar or array
    VectorXd mass_vec;
    if (py::isinstance<py::float_>(mass) || py::isinstance<py::int_>(mass)) {
        mass_vec = VectorXd::Constant(1, mass.cast<double>());
    } else {
        py::array_t<double> mass_arr = py::cast<py::array_t<double>>(mass);
        auto m_r = mass_arr.unchecked<1>();
        if ((int)m_r.shape(0) != n && m_r.shape(0) != 1) throw std::runtime_error("mass length mismatch");
        if (m_r.shape(0) == 1) mass_vec = VectorXd::Constant(1, m_r(0));
        else {
            mass_vec = VectorXd(n);
            for (int i = 0; i < n; ++i) mass_vec[i] = m_r(i);
        }
    }

    int n_step = int(std::round(t_total / dt));
    if (n_step < 1) n_step = 1;

    return sprk_core(force_func, mass_vec, q0_vec, p0_vec, dt, n_step, true);
}

PYBIND11_MODULE(sprk, m) {
    m.doc() = "Symplectic Partitioned Runge-Kutta (8-stage) SPRK integrator (Eigen + pybind11)";

    m.def("sprk_with_pyforce", &sprk_with_pyforce,
          py::arg("force_func"),
          py::arg("mass"),
          py::arg("q0"),
          py::arg("p0"),
          py::arg("t_total"),
          py::arg("dt"),
          "Run SPRK using a Python callable force_func(q) -> f(q). Returns (t, q, p).");
}

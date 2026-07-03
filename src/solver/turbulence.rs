//! k-epsilon turbulence model
//!
//! Standard k-ε model equations:
//!   ∂k/∂t + u·∇k = ∇·[(ν + νt/σk)∇k] + P - ε
//!   ∂ε/∂t + u·∇ε = ∇·[(ν + νt/σε)∇ε] + C1ε·ε/k·P - C2ε·ε²/k
//!
//! where:
//!   νt = Cμ·k²/ε  (turbulent viscosity)
//!   P = νt·|S|²   (production term, S = strain rate tensor)

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// k-epsilon model constants (standard values)
pub struct KEpsilonConstants {
    pub c_mu: f64,      // 0.09
    pub c_1e: f64,      // 1.44
    pub c_2e: f64,      // 1.92
    pub sigma_k: f64,   // 1.0
    pub sigma_e: f64,   // 1.3
}

impl Default for KEpsilonConstants {
    fn default() -> Self {
        Self {
            c_mu: 0.09,
            c_1e: 1.44,
            c_2e: 1.92,
            sigma_k: 1.0,
            sigma_e: 1.3,
        }
    }
}

/// k-epsilon turbulence model
pub struct KEpsilonModel {
    pub constants: KEpsilonConstants,
    /// Minimum k value (prevents division by zero)
    pub k_min: f64,
    /// Minimum epsilon value
    pub eps_min: f64,
    /// Maximum turbulent viscosity ratio (ν_t / ν)
    pub nu_t_ratio_max: f64,
}

impl Default for KEpsilonModel {
    fn default() -> Self {
        Self {
            constants: KEpsilonConstants::default(),
            k_min: 1e-10,
            eps_min: 1e-10,
            nu_t_ratio_max: 1e5,
        }
    }
}

impl KEpsilonModel {
    /// Create new k-epsilon model with default constants
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute strain rate magnitude |S| at cell center
    /// |S|² = 2·Sij·Sij where Sij = 0.5·(∂ui/∂xj + ∂uj/∂xi)
    pub fn strain_rate_magnitude(&self, fields: &Fields, mesh: &Mesh, i: usize, j: usize) -> f64 {
        // Velocity gradients
        let du_dx = if i == 0 {
            (fields.u[[i + 1, j]] - fields.u[[i, j]]) / mesh.dx
        } else if i == mesh.nx - 1 {
            (fields.u[[i + 1, j]] - fields.u[[i, j]]) / mesh.dx
        } else {
            (fields.u[[i + 1, j]] - fields.u[[i, j]]) / mesh.dx
        };

        let du_dy = if j == 0 {
            let u_top = 0.5 * (fields.u[[i, j + 1]] + fields.u[[i + 1, j + 1]]);
            let u_bot = 0.5 * (fields.u[[i, j]] + fields.u[[i + 1, j]]);
            (u_top - u_bot) / mesh.dy
        } else if j == mesh.ny - 1 {
            let u_top = 0.5 * (fields.u[[i, j]] + fields.u[[i + 1, j]]);
            let u_bot = 0.5 * (fields.u[[i, j - 1]] + fields.u[[i + 1, j - 1]]);
            (u_top - u_bot) / mesh.dy
        } else {
            let u_top = 0.5 * (fields.u[[i, j + 1]] + fields.u[[i + 1, j + 1]]);
            let u_bot = 0.5 * (fields.u[[i, j]] + fields.u[[i + 1, j]]);
            (u_top - u_bot) / mesh.dy
        };

        let dv_dx = if i == 0 {
            let v_right = 0.5 * (fields.v[[i + 1, j]] + fields.v[[i + 1, j + 1]]);
            let v_left = 0.5 * (fields.v[[i, j]] + fields.v[[i, j + 1]]);
            (v_right - v_left) / mesh.dx
        } else if i == mesh.nx - 1 {
            let v_right = 0.5 * (fields.v[[i, j]] + fields.v[[i, j + 1]]);
            let v_left = 0.5 * (fields.v[[i - 1, j]] + fields.v[[i - 1, j + 1]]);
            (v_right - v_left) / mesh.dx
        } else {
            let v_right = 0.5 * (fields.v[[i + 1, j]] + fields.v[[i + 1, j + 1]]);
            let v_left = 0.5 * (fields.v[[i, j]] + fields.v[[i, j + 1]]);
            (v_right - v_left) / mesh.dx
        };

        let dv_dy = if j == 0 {
            (fields.v[[i, j + 1]] - fields.v[[i, j]]) / mesh.dy
        } else if j == mesh.ny - 1 {
            (fields.v[[i, j + 1]] - fields.v[[i, j]]) / mesh.dy
        } else {
            (fields.v[[i, j + 1]] - fields.v[[i, j]]) / mesh.dy
        };

        // Strain rate tensor components (2D):
        // S11 = du/dx, S22 = dv/dy, S12 = S21 = 0.5*(du/dy + dv/dx)
        let s11 = du_dx;
        let s22 = dv_dy;
        let s12 = 0.5 * (du_dy + dv_dx);

        // |S|² = 2*(S11² + S22² + 2*S12²) for 2D
        let s_mag_sq = 2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12);
        s_mag_sq.sqrt()
    }

    /// Compute turbulent viscosity: νt = Cμ·k²/ε
    pub fn compute_nu_t(
        &self,
        k: &Array2<f64>,
        epsilon: &Array2<f64>,
        nu: &Array2<f64>,
    ) -> Array2<f64> {
        let shape = k.raw_dim();
        let mut nu_t = Array2::zeros(shape);

        for j in 0..shape[1] {
            for i in 0..shape[0] {
                let k_val = k[[i, j]].max(self.k_min);
                let eps_val = epsilon[[i, j]].max(self.eps_min);
                
                // νt = Cμ·k²/ε
                let mut nu_t_val = self.constants.c_mu * k_val * k_val / eps_val;
                
                // Limit turbulent viscosity ratio
                let nu_val = nu[[i, j]].max(1e-10);
                let max_nu_t = self.nu_t_ratio_max * nu_val;
                nu_t_val = nu_t_val.min(max_nu_t);
                
                nu_t[[i, j]] = nu_t_val;
            }
        }

        nu_t
    }

    /// Advance k and epsilon by one time step
    /// Returns (k_new, epsilon_new)
    pub fn advance(
        &self,
        fields: &Fields,
        k: &Array2<f64>,
        epsilon: &Array2<f64>,
        nu_t: &Array2<f64>,
        mesh: &Mesh,
        dt: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        let nx = mesh.nx;
        let ny = mesh.ny;
        
        let mut k_new = k.clone();
        let mut eps_new = epsilon.clone();
        
        let c = &self.constants;

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // Skip solid cells
                if fields.is_solid(i, j) {
                    continue;
                }
                
                // Skip air cells (VOF < threshold)
                if fields.vof[[i, j]] < 0.1 {
                    k_new[[i, j]] = self.k_min;
                    eps_new[[i, j]] = self.eps_min;
                    continue;
                }

                let k_val = k[[i, j]].max(self.k_min);
                let eps_val = epsilon[[i, j]].max(self.eps_min);
                let nu_val = fields.nu[[i, j]];
                let nu_t_val = nu_t[[i, j]];

                // === Advection (upwind) ===
                let uc = fields.u_at_center(i, j);
                let vc = fields.v_at_center(i, j);

                // k advection
                let dk_dx = if uc > 0.0 {
                    (k[[i, j]] - k[[i - 1, j]]) / mesh.dx
                } else {
                    (k[[i + 1, j]] - k[[i, j]]) / mesh.dx
                };
                let dk_dy = if vc > 0.0 {
                    (k[[i, j]] - k[[i, j - 1]]) / mesh.dy
                } else {
                    (k[[i, j + 1]] - k[[i, j]]) / mesh.dy
                };
                let adv_k = uc * dk_dx + vc * dk_dy;

                // epsilon advection
                let de_dx = if uc > 0.0 {
                    (epsilon[[i, j]] - epsilon[[i - 1, j]]) / mesh.dx
                } else {
                    (epsilon[[i + 1, j]] - epsilon[[i, j]]) / mesh.dx
                };
                let de_dy = if vc > 0.0 {
                    (epsilon[[i, j]] - epsilon[[i, j - 1]]) / mesh.dy
                } else {
                    (epsilon[[i, j + 1]] - epsilon[[i, j]]) / mesh.dy
                };
                let adv_e = uc * de_dx + vc * de_dy;

                // === Diffusion ===
                let diff_k_coeff = nu_val + nu_t_val / c.sigma_k;
                let diff_e_coeff = nu_val + nu_t_val / c.sigma_e;

                // k diffusion (Laplacian)
                let d2k_dx2 = (k[[i + 1, j]] - 2.0 * k[[i, j]] + k[[i - 1, j]]) / (mesh.dx * mesh.dx);
                let d2k_dy2 = (k[[i, j + 1]] - 2.0 * k[[i, j]] + k[[i, j - 1]]) / (mesh.dy * mesh.dy);
                let diff_k = diff_k_coeff * (d2k_dx2 + d2k_dy2);

                // epsilon diffusion
                let d2e_dx2 = (epsilon[[i + 1, j]] - 2.0 * epsilon[[i, j]] + epsilon[[i - 1, j]]) / (mesh.dx * mesh.dx);
                let d2e_dy2 = (epsilon[[i, j + 1]] - 2.0 * epsilon[[i, j]] + epsilon[[i, j - 1]]) / (mesh.dy * mesh.dy);
                let diff_e = diff_e_coeff * (d2e_dx2 + d2e_dy2);

                // === Production ===
                let s_mag = self.strain_rate_magnitude(fields, mesh, i, j);
                let production = nu_t_val * s_mag * s_mag;

                // === Source terms ===
                // k equation: ∂k/∂t = -adv + diff + P - ε
                let dk_dt = -adv_k + diff_k + production - eps_val;
                
                // ε equation: ∂ε/∂t = -adv + diff + C1ε·ε/k·P - C2ε·ε²/k
                let de_dt = -adv_e + diff_e 
                    + c.c_1e * eps_val / k_val * production 
                    - c.c_2e * eps_val * eps_val / k_val;

                // === Time integration (explicit Euler) ===
                k_new[[i, j]] = (k_val + dt * dk_dt).max(self.k_min);
                eps_new[[i, j]] = (eps_val + dt * de_dt).max(self.eps_min);
            }
        }

        // Boundary conditions (zero gradient)
        self.apply_bc(&mut k_new, &mut eps_new, mesh, fields);

        (k_new, eps_new)
    }

    /// Apply boundary conditions for k and epsilon
    fn apply_bc(
        &self,
        k: &mut Array2<f64>,
        epsilon: &mut Array2<f64>,
        mesh: &Mesh,
        fields: &Fields,
    ) {
        let nx = mesh.nx;
        let ny = mesh.ny;

        // Left and right boundaries (zero gradient)
        for j in 0..ny {
            k[[0, j]] = k[[1, j]];
            k[[nx - 1, j]] = k[[nx - 2, j]];
            epsilon[[0, j]] = epsilon[[1, j]];
            epsilon[[nx - 1, j]] = epsilon[[nx - 2, j]];
        }

        // Bottom: wall function approach - set k=0 at wall, epsilon from equilibrium
        for i in 0..nx {
            // Find first fluid cell from bottom
            let mut j_first_fluid = 0;
            for j in 0..ny {
                if !fields.is_solid(i, j) {
                    j_first_fluid = j;
                    break;
                }
            }
            
            if j_first_fluid > 0 {
                // Wall adjacent cell - use wall functions
                let k_wall = self.k_min;
                let y_plus = mesh.dy * 0.5; // Distance to wall
                let u_mag = fields.velocity_magnitude(i, j_first_fluid);
                let nu_val = fields.nu[[i, j_first_fluid]].max(1e-10);
                
                // Approximate wall shear stress
                let tau_w = nu_val * u_mag / y_plus;
                let u_tau = tau_w.sqrt();
                
                // Equilibrium epsilon at wall: ε = Cμ^0.75 * k^1.5 / (κ*y)
                let kappa = 0.41; // von Karman constant
                let k_near_wall = k[[i, j_first_fluid]].max(self.k_min);
                let eps_wall = self.constants.c_mu.powf(0.75) * k_near_wall.powf(1.5) / (kappa * y_plus);
                
                // Set values at solid cells below
                for j in 0..j_first_fluid {
                    k[[i, j]] = k_wall;
                    epsilon[[i, j]] = eps_wall.max(self.eps_min);
                }
            } else {
                // Bottom boundary without bathymetry
                k[[i, 0]] = k[[i, 1]];
                epsilon[[i, 0]] = epsilon[[i, 1]];
            }
        }

        // Top boundary (zero gradient - open to atmosphere)
        for i in 0..nx {
            k[[i, ny - 1]] = k[[i, ny - 2]];
            epsilon[[i, ny - 1]] = epsilon[[i, ny - 2]];
        }

        // Ensure minimum values
        for val in k.iter_mut() {
            *val = val.max(self.k_min);
        }
        for val in epsilon.iter_mut() {
            *val = val.max(self.eps_min);
        }
    }

    /// Initialize k and epsilon from turbulence intensity and length scale
    pub fn initialize(
        &self,
        fields: &Fields,
        mesh: &Mesh,
        turbulence_intensity: f64,  // Typically 0.01-0.1 (1%-10%)
        length_scale: f64,          // Typically 0.07 * hydraulic diameter
    ) -> (Array2<f64>, Array2<f64>) {
        let nx = mesh.nx;
        let ny = mesh.ny;
        
        let mut k = Array2::zeros((nx, ny));
        let mut epsilon = Array2::zeros((nx, ny));

        for j in 0..ny {
            for i in 0..nx {
                let u_mag = fields.velocity_magnitude(i, j).max(0.01);
                
                // k = 1.5 * (U * I)²
                let k_val = 1.5 * (u_mag * turbulence_intensity).powi(2);
                
                // ε = Cμ^0.75 * k^1.5 / L
                let eps_val = self.constants.c_mu.powf(0.75) * k_val.powf(1.5) / length_scale;

                k[[i, j]] = k_val.max(self.k_min);
                epsilon[[i, j]] = eps_val.max(self.eps_min);
            }
        }

        (k, epsilon)
    }
}

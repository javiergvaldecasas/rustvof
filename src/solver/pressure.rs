 //! Pressure Poisson solver using Jacobi iteration
//!
//! Solves: ∇²p = RHS
//! Where RHS = (ρ/Δt) ∇·u*

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::fields::Fields;

// Note: Fields is used in compute_rhs methods

/// Pressure solver using Jacobi iteration
pub struct PressureSolver {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Over-relaxation factor (1.0 = Jacobi, >1 = SOR)
    pub omega: f64,
}

impl Default for PressureSolver {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tolerance: 1e-6,
            omega: 1.5, // SOR with ω=1.5 for faster convergence
        }
    }
}

impl PressureSolver {
    /// Solve the pressure Poisson equation with VOF-aware boundary conditions
    /// 
    /// ∇²p = rhs
    /// 
    /// - Neumann BC (∂p/∂n = 0) on solid walls
    /// - Dirichlet BC (p = 0) at free surface (where VOF < 0.5 = air)
    /// 
    /// Returns (converged, iterations, residual)
    pub fn solve(
        &self,
        p: &mut Array2<f64>,
        rhs: &Array2<f64>,
        mesh: &Mesh,
    ) -> (bool, usize, f64) {
        let dx2 = mesh.dx * mesh.dx;
        let dy2 = mesh.dy * mesh.dy;
        let factor = 2.0 * (1.0 / dx2 + 1.0 / dy2);
        
        let mut residual: f64 = 0.0;
        
        for iter in 0..self.max_iter {
            residual = 0.0;
            
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    // Get neighbor values with boundary conditions
                    let p_left = if i > 0 { p[[i - 1, j]] } else { p[[i, j]] };    // Neumann
                    let p_right = if i < mesh.nx - 1 { p[[i + 1, j]] } else { p[[i, j]] }; // Neumann
                    let p_bottom = if j > 0 { p[[i, j - 1]] } else { p[[i, j]] };  // Neumann (floor)
                    
                    // Top boundary: Dirichlet p=0 at domain top OR at free surface
                    let p_top = if j < mesh.ny - 1 {
                        p[[i, j + 1]]
                    } else {
                        // At domain top - atmospheric pressure
                        0.0
                    };
                    
                    // Jacobi/SOR update
                    let p_new = (
                        (p_left + p_right) / dx2 +
                        (p_bottom + p_top) / dy2 -
                        rhs[[i, j]]
                    ) / factor;
                    
                    // SOR relaxation
                    let p_old = p[[i, j]];
                    p[[i, j]] = p_old + self.omega * (p_new - p_old);
                    
                    // Accumulate residual
                    let res = (p_new - p_old).abs();
                    residual = residual.max(res);
                }
            }
            
            // Check convergence
            if residual < self.tolerance {
                return (true, iter + 1, residual);
            }
        }
        
        (false, self.max_iter, residual)
    }
    
    /// Solve pressure equation with variable density (two-phase flow)
    /// 
    /// Solves: ∇·(1/ρ ∇p) = ∇·u*/Δt
    /// 
    /// With:
    /// - Neumann BC on solid walls
    /// - Dirichlet p=0 at free surface (VOF < 0.5)
    pub fn solve_with_vof(
        &self,
        p: &mut Array2<f64>,
        rhs: &Array2<f64>,
        vof: &Array2<f64>,
        mesh: &Mesh,
    ) -> (bool, usize, f64) {
        let dx = mesh.dx;
        let dy = mesh.dy;
        
        // Reference density for scaling
        const RHO_WATER: f64 = 1000.0;
        const RHO_AIR: f64 = 1.225;
        
        // Helper to get density from VOF
        let get_rho = |v: f64| -> f64 {
            v * RHO_WATER + (1.0 - v) * RHO_AIR
        };
        
        let mut residual: f64 = 0.0;
        
        // Set p=0 in pure air cells
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                if vof[[i, j]] < 0.1 {  // Pure air
                    p[[i, j]] = 0.0;
                }
            }
        }
        
        for iter in 0..self.max_iter {
            residual = 0.0;
            
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    let vof_c = vof[[i, j]];
                    
                    // Skip pure air cells
                    if vof_c < 0.1 {
                        continue;
                    }
                    
                    let rho_c = get_rho(vof_c);
                    
                    // Compute 1/ρ at cell faces (harmonic average for conservation)
                    let mut sum = 0.0;
                    let mut coeff = 0.0;
                    
                    // Left face (i-1/2)
                    let (p_left, rho_left) = if i > 0 {
                        let vof_l = vof[[i - 1, j]];
                        if vof_l < 0.1 {
                            // Air: Dirichlet p=0
                            (0.0, get_rho(0.5 * (vof_c + vof_l)))
                        } else {
                            (p[[i - 1, j]], get_rho(0.5 * (vof_c + vof_l)))
                        }
                    } else {
                        // Wall: Neumann ∂p/∂n = 0
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_left = 1.0 / rho_left;
                    sum += inv_rho_left * p_left / (dx * dx);
                    coeff += inv_rho_left / (dx * dx);
                    
                    // Right face (i+1/2)
                    let (p_right, rho_right) = if i < mesh.nx - 1 {
                        let vof_r = vof[[i + 1, j]];
                        if vof_r < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_r)))
                        } else {
                            (p[[i + 1, j]], get_rho(0.5 * (vof_c + vof_r)))
                        }
                    } else {
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_right = 1.0 / rho_right;
                    sum += inv_rho_right * p_right / (dx * dx);
                    coeff += inv_rho_right / (dx * dx);
                    
                    // Bottom face (j-1/2)
                    let (p_bottom, rho_bottom) = if j > 0 {
                        let vof_b = vof[[i, j - 1]];
                        if vof_b < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_b)))
                        } else {
                            (p[[i, j - 1]], get_rho(0.5 * (vof_c + vof_b)))
                        }
                    } else {
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_bottom = 1.0 / rho_bottom;
                    sum += inv_rho_bottom * p_bottom / (dy * dy);
                    coeff += inv_rho_bottom / (dy * dy);
                    
                    // Top face (j+1/2)
                    let (p_top, rho_top) = if j < mesh.ny - 1 {
                        let vof_t = vof[[i, j + 1]];
                        if vof_t < 0.1 {
                            // Free surface: p=0
                            (0.0, get_rho(0.5 * (vof_c + vof_t)))
                        } else {
                            (p[[i, j + 1]], get_rho(0.5 * (vof_c + vof_t)))
                        }
                    } else {
                        // Domain top: p=0
                        (0.0, rho_c)
                    };
                    let inv_rho_top = 1.0 / rho_top;
                    sum += inv_rho_top * p_top / (dy * dy);
                    coeff += inv_rho_top / (dy * dy);
                    
                    // Solve for p
                    if coeff.abs() < 1e-10 {
                        continue;  // Skip degenerate cells
                    }
                    let p_new = (sum - rhs[[i, j]]) / coeff;
                    
                    // SOR relaxation
                    let p_old = p[[i, j]];
                    p[[i, j]] = p_old + self.omega * (p_new - p_old);
                    
                    // Accumulate residual
                    let res = (p_new - p_old).abs();
                    residual = residual.max(res);
                }
            }
            
            // Check convergence
            if residual < self.tolerance {
                return (true, iter + 1, residual);
            }
        }
        
        (false, self.max_iter, residual)
    }

    /// Solve pressure equation with solid mask (for moving paddle)
    /// 
    /// Solid cells are excluded from the solve (p = 0 in solid region)
    pub fn solve_with_solid_mask(
        &self,
        p: &mut Array2<f64>,
        rhs: &Array2<f64>,
        vof: &Array2<f64>,
        solid_mask: &[Vec<bool>],
        mesh: &Mesh,
    ) -> (bool, usize, f64) {
        let dx = mesh.dx;
        let dy = mesh.dy;
        
        const RHO_WATER: f64 = 1000.0;
        const RHO_AIR: f64 = 1.225;
        
        let get_rho = |v: f64| -> f64 {
            v * RHO_WATER + (1.0 - v) * RHO_AIR
        };
        
        let mut residual: f64 = 0.0;
        
        // Set p=0 in solid cells and pure air cells
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                if solid_mask[i][j] || vof[[i, j]] < 0.1 {
                    p[[i, j]] = 0.0;
                }
            }
        }
        
        for iter in 0..self.max_iter {
            residual = 0.0;
            
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    // Skip solid cells
                    if solid_mask[i][j] {
                        continue;
                    }
                    
                    let vof_c = vof[[i, j]];
                    
                    // Skip pure air cells
                    if vof_c < 0.1 {
                        continue;
                    }
                    
                    let rho_c = get_rho(vof_c);
                    
                    let mut sum = 0.0;
                    let mut coeff = 0.0;
                    
                    // Left face
                    let (p_left, rho_left) = if i > 0 && !solid_mask[i - 1][j] {
                        let vof_l = vof[[i - 1, j]];
                        if vof_l < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_l)))
                        } else {
                            (p[[i - 1, j]], get_rho(0.5 * (vof_c + vof_l)))
                        }
                    } else {
                        // Wall or solid: Neumann
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_left = 1.0 / rho_left;
                    sum += inv_rho_left * p_left / (dx * dx);
                    coeff += inv_rho_left / (dx * dx);
                    
                    // Right face
                    let (p_right, rho_right) = if i < mesh.nx - 1 && !solid_mask[i + 1][j] {
                        let vof_r = vof[[i + 1, j]];
                        if vof_r < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_r)))
                        } else {
                            (p[[i + 1, j]], get_rho(0.5 * (vof_c + vof_r)))
                        }
                    } else {
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_right = 1.0 / rho_right;
                    sum += inv_rho_right * p_right / (dx * dx);
                    coeff += inv_rho_right / (dx * dx);
                    
                    // Bottom face
                    let (p_bottom, rho_bottom) = if j > 0 && !solid_mask[i][j - 1] {
                        let vof_b = vof[[i, j - 1]];
                        if vof_b < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_b)))
                        } else {
                            (p[[i, j - 1]], get_rho(0.5 * (vof_c + vof_b)))
                        }
                    } else {
                        (p[[i, j]], rho_c)
                    };
                    let inv_rho_bottom = 1.0 / rho_bottom;
                    sum += inv_rho_bottom * p_bottom / (dy * dy);
                    coeff += inv_rho_bottom / (dy * dy);
                    
                    // Top face
                    let (p_top, rho_top) = if j < mesh.ny - 1 && !solid_mask[i][j + 1] {
                        let vof_t = vof[[i, j + 1]];
                        if vof_t < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_t)))
                        } else {
                            (p[[i, j + 1]], get_rho(0.5 * (vof_c + vof_t)))
                        }
                    } else {
                        (0.0, rho_c)
                    };
                    let inv_rho_top = 1.0 / rho_top;
                    sum += inv_rho_top * p_top / (dy * dy);
                    coeff += inv_rho_top / (dy * dy);
                    
                    if coeff.abs() < 1e-10 {
                        continue;
                    }
                    let p_new = (sum - rhs[[i, j]]) / coeff;
                    
                    let p_old = p[[i, j]];
                    p[[i, j]] = p_old + self.omega * (p_new - p_old);
                    
                    let res = (p_new - p_old).abs();
                    residual = residual.max(res);
                }
            }
            
            if residual < self.tolerance {
                return (true, iter + 1, residual);
            }
        }
        
        (false, self.max_iter, residual)
    }

    /// Solve pressure equation using Fields struct directly
    /// 
    /// Uses fields.is_solid() for solid cell detection and proper BCs:
    /// - Solid cells: skip (p unchanged)
    /// - Air cells (VOF < 0.1): Dirichlet p=0
    /// - Fluid cells: solve with Neumann BC at solid interfaces
    pub fn solve_with_fields(
        &self,
        fields: &mut Fields,
        rhs: &Array2<f64>,
        mesh: &Mesh,
    ) -> (bool, usize, f64) {
        let dx = mesh.dx;
        let dy = mesh.dy;
        
        const RHO_WATER: f64 = 1000.0;
        const RHO_AIR: f64 = 1.225;
        
        let get_rho = |v: f64| -> f64 {
            v * RHO_WATER + (1.0 - v) * RHO_AIR
        };
        
        let mut residual: f64 = 0.0;
        
        // Set p=0 in solid cells and pure air cells
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                if fields.is_solid(i, j) || fields.vof[[i, j]] < 0.1 {
                    fields.p[[i, j]] = 0.0;
                }
            }
        }
        
        for iter in 0..self.max_iter {
            residual = 0.0;
            
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    // Skip solid cells
                    if fields.is_solid(i, j) {
                        continue;
                    }
                    
                    let vof_c = fields.vof[[i, j]];
                    
                    // Skip pure air cells
                    if vof_c < 0.1 {
                        continue;
                    }
                    
                    let rho_c = get_rho(vof_c);
                    
                    let mut sum = 0.0;
                    let mut coeff = 0.0;
                    
                    // Left face - check if neighbor is solid
                    let left_is_solid = i == 0 || fields.is_solid(i - 1, j);
                    let (p_left, rho_left) = if !left_is_solid {
                        let vof_l = fields.vof[[i - 1, j]];
                        if vof_l < 0.1 {
                            // Air: Dirichlet p=0
                            (0.0, get_rho(0.5 * (vof_c + vof_l)))
                        } else {
                            (fields.p[[i - 1, j]], get_rho(0.5 * (vof_c + vof_l)))
                        }
                    } else {
                        // Solid wall: Neumann ∂p/∂n = 0
                        (fields.p[[i, j]], rho_c)
                    };
                    let inv_rho_left = 1.0 / rho_left;
                    sum += inv_rho_left * p_left / (dx * dx);
                    coeff += inv_rho_left / (dx * dx);
                    
                    // Right face
                    let right_is_solid = i >= mesh.nx - 1 || fields.is_solid(i + 1, j);
                    let (p_right, rho_right) = if !right_is_solid {
                        let vof_r = fields.vof[[i + 1, j]];
                        if vof_r < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_r)))
                        } else {
                            (fields.p[[i + 1, j]], get_rho(0.5 * (vof_c + vof_r)))
                        }
                    } else {
                        (fields.p[[i, j]], rho_c)
                    };
                    let inv_rho_right = 1.0 / rho_right;
                    sum += inv_rho_right * p_right / (dx * dx);
                    coeff += inv_rho_right / (dx * dx);
                    
                    // Bottom face - CRITICAL for bathymetry
                    let bottom_is_solid = j == 0 || fields.is_solid(i, j - 1);
                    let (p_bottom, rho_bottom) = if !bottom_is_solid {
                        let vof_b = fields.vof[[i, j - 1]];
                        if vof_b < 0.1 {
                            (0.0, get_rho(0.5 * (vof_c + vof_b)))
                        } else {
                            (fields.p[[i, j - 1]], get_rho(0.5 * (vof_c + vof_b)))
                        }
                    } else {
                        // Solid floor/ramp: Neumann ∂p/∂n = 0
                        (fields.p[[i, j]], rho_c)
                    };
                    let inv_rho_bottom = 1.0 / rho_bottom;
                    sum += inv_rho_bottom * p_bottom / (dy * dy);
                    coeff += inv_rho_bottom / (dy * dy);
                    
                    // Top face
                    let top_is_solid = j >= mesh.ny - 1 || fields.is_solid(i, j + 1);
                    let (p_top, rho_top) = if !top_is_solid {
                        let vof_t = fields.vof[[i, j + 1]];
                        if vof_t < 0.1 {
                            // Free surface: p=0
                            (0.0, get_rho(0.5 * (vof_c + vof_t)))
                        } else {
                            (fields.p[[i, j + 1]], get_rho(0.5 * (vof_c + vof_t)))
                        }
                    } else {
                        // Domain top: p=0 (open boundary)
                        (0.0, rho_c)
                    };
                    let inv_rho_top = 1.0 / rho_top;
                    sum += inv_rho_top * p_top / (dy * dy);
                    coeff += inv_rho_top / (dy * dy);
                    
                    // Skip degenerate cells
                    if coeff.abs() < 1e-10 {
                        continue;
                    }
                    
                    // Solve for p
                    let p_new = (sum - rhs[[i, j]]) / coeff;
                    
                    // SOR relaxation
                    let p_old = fields.p[[i, j]];
                    fields.p[[i, j]] = p_old + self.omega * (p_new - p_old);
                    
                    let res = (p_new - p_old).abs();
                    residual = residual.max(res);
                }
            }
            
            if residual < self.tolerance {
                return (true, iter + 1, residual);
            }
        }
        
        (false, self.max_iter, residual)
    }

    /// Compute RHS of pressure equation from intermediate velocity
    /// RHS = (1/Δt) ∇·u*
    pub fn compute_rhs(
        fields: &Fields,
        mesh: &Mesh,
        dt: f64,
    ) -> Array2<f64> {
        let mut rhs = Array2::zeros((mesh.nx, mesh.ny));
        
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let div = fields.divergence(mesh, i, j);
                rhs[[i, j]] = div / dt;
            }
        }
        
        rhs
    }

    /// Compute RHS with density weighting for two-phase flow
    /// RHS = (ρ/Δt) ∇·u*
    pub fn compute_rhs_weighted(
        fields: &Fields,
        mesh: &Mesh,
        dt: f64,
    ) -> Array2<f64> {
        let mut rhs = Array2::zeros((mesh.nx, mesh.ny));
        
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let div = fields.divergence(mesh, i, j);
                let rho = fields.rho[[i, j]];
                rhs[[i, j]] = rho * div / dt;
            }
        }
        
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;

    #[test]
    fn test_pressure_solver_zero_rhs() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut p = Array2::zeros((mesh.nx, mesh.ny));
        let rhs = Array2::zeros((mesh.nx, mesh.ny));
        
        let solver = PressureSolver::default();
        let (converged, _, _) = solver.solve(&mut p, &rhs, &mesh);
        
        assert!(converged);
        // With zero RHS and homogeneous BCs, solution should be zero
        for &val in p.iter() {
            assert!(val.abs() < 1e-10);
        }
    }
}

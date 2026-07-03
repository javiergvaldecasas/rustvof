//! Field storage for simulation variables
//!
//! Stores VOF, pressure, and velocity fields on the staggered grid.

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::config::InitialConditionConfig;
use crate::cell_type::CellType;

/// All simulation fields
#[derive(Debug, Clone)]
pub struct Fields {
    /// Volume of Fluid fraction (0=air, 1=water) at cell centers
    pub vof: Array2<f64>,
    /// Pressure at cell centers
    pub p: Array2<f64>,
    /// u-velocity (x-direction) at vertical faces
    pub u: Array2<f64>,
    /// v-velocity (y-direction) at horizontal faces
    pub v: Array2<f64>,
    /// Density at cell centers (derived from VOF)
    pub rho: Array2<f64>,
    /// Kinematic viscosity at cell centers (derived from VOF)
    pub nu: Array2<f64>,
    /// Cell type (fluid/solid/porous) at cell centers
    pub cell_type: Array2<CellType>,
    /// Fractional aperture for u-faces (x-direction), 0=blocked, 1=open
    /// ar[[i,j]] is aperture of face between cell (i-1,j) and (i,j)
    pub ar: Array2<f64>,
    /// Fractional aperture for v-faces (y-direction), 0=blocked, 1=open
    /// at[[i,j]] is aperture of face between cell (i,j-1) and (i,j)
    pub at: Array2<f64>,
    /// Cell aperture (fraction of cell open to flow), 0=solid, 1=fluid
    pub ac: Array2<f64>,
    /// Turbulent kinetic energy k (for k-ε model)
    pub k: Array2<f64>,
    /// Turbulent dissipation rate ε (for k-ε model)
    pub epsilon: Array2<f64>,
    /// Turbulent viscosity νt (computed from k and ε)
    pub nu_t: Array2<f64>,
}

impl Fields {
    /// Create new fields initialized to zero
    pub fn new(mesh: &Mesh) -> Self {
        let (nu_x, nu_y) = mesh.num_u();
        let (nv_x, nv_y) = mesh.num_v();
        
        Self {
            vof: Array2::zeros((mesh.nx, mesh.ny)),
            p: Array2::zeros((mesh.nx, mesh.ny)),
            u: Array2::zeros((nu_x, nu_y)),
            v: Array2::zeros((nv_x, nv_y)),
            rho: Array2::zeros((mesh.nx, mesh.ny)),
            nu: Array2::zeros((mesh.nx, mesh.ny)),
            cell_type: Array2::default((mesh.nx, mesh.ny)),  // All Fluid by default
            ar: Array2::ones((nu_x, nu_y)),  // All faces open by default
            at: Array2::ones((nv_x, nv_y)),  // All faces open by default
            ac: Array2::ones((mesh.nx, mesh.ny)),  // All cells open by default
            k: Array2::zeros((mesh.nx, mesh.ny)),       // Turbulent kinetic energy
            epsilon: Array2::zeros((mesh.nx, mesh.ny)), // Turbulent dissipation
            nu_t: Array2::zeros((mesh.nx, mesh.ny)),    // Turbulent viscosity
        }
    }
    
    /// Set cell type for a region
    pub fn set_cell_type_rect(
        &mut self,
        mesh: &Mesh,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        cell_type: CellType,
    ) {
        for j in 0..mesh.ny {
            let y = mesh.y_center(j);
            for i in 0..mesh.nx {
                let x = mesh.x_center(i);
                if x >= x_min && x <= x_max && y >= y_min && y <= y_max {
                    self.cell_type[[i, j]] = cell_type;
                }
            }
        }
    }
    
    /// Set cells as solid where x < x_paddle (for moving paddle)
    pub fn set_solid_behind_paddle(&mut self, mesh: &Mesh, x_paddle: f64) {
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let x = mesh.x_center(i);
                if x < x_paddle {
                    // Only set to Paddle if not already a permanent solid (bathymetry)
                    if !self.cell_type[[i, j]].is_permanent_solid() {
                        self.cell_type[[i, j]] = CellType::Paddle;
                    }
                } else if self.cell_type[[i, j]].is_paddle() {
                    // Only convert back to fluid if it was Paddle (not permanent Solid)
                    self.cell_type[[i, j]] = CellType::Fluid;
                }
            }
        }
    }
    
    /// Check if cell is fluid
    pub fn is_fluid(&self, i: usize, j: usize) -> bool {
        self.cell_type[[i, j]].is_fluid()
    }
    
    /// Check if cell is solid
    pub fn is_solid(&self, i: usize, j: usize) -> bool {
        self.cell_type[[i, j]].is_solid()
    }
    
    /// Check if cell allows flow (fluid or porous)
    pub fn allows_flow(&self, i: usize, j: usize) -> bool {
        self.cell_type[[i, j]].allows_flow()
    }
    
    /// Set cell as solid
    pub fn set_solid(&mut self, i: usize, j: usize) {
        self.cell_type[[i, j]] = CellType::Solid;
    }
    
    /// Set cell as fluid
    pub fn set_fluid(&mut self, i: usize, j: usize) {
        self.cell_type[[i, j]] = CellType::Fluid;
    }

    /// Initialize fields with initial condition (water level)
    pub fn initialize(
        &mut self,
        mesh: &Mesh,
        ic: &InitialConditionConfig,
        rho_water: f64,
        rho_air: f64,
        nu_water: f64,
        nu_air: f64,
    ) {
        // Set VOF based on water level
        for j in 0..mesh.ny {
            let y_bottom = j as f64 * mesh.dy;
            let y_top = (j + 1) as f64 * mesh.dy;
            
            for i in 0..mesh.nx {
                // Calculate VOF fraction for this cell
                let vof_val = if y_top <= ic.water_level {
                    // Cell completely below water level
                    1.0
                } else if y_bottom >= ic.water_level {
                    // Cell completely above water level
                    0.0
                } else {
                    // Cell partially filled - linear interpolation
                    (ic.water_level - y_bottom) / mesh.dy
                };
                
                self.vof[[i, j]] = vof_val;
            }
        }

        // Update density and viscosity based on VOF
        self.update_properties(rho_water, rho_air, nu_water, nu_air);
    }

    /// Update density and viscosity from VOF field
    pub fn update_properties(
        &mut self,
        rho_water: f64,
        rho_air: f64,
        nu_water: f64,
        nu_air: f64,
    ) {
        for ((i, j), &vof) in self.vof.indexed_iter() {
            self.rho[[i, j]] = vof * rho_water + (1.0 - vof) * rho_air;
            self.nu[[i, j]] = vof * nu_water + (1.0 - vof) * nu_air;
        }
    }

    /// Get maximum velocity magnitude (for CFL calculation)
    pub fn max_velocity(&self) -> f64 {
        let max_u = self.u.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let max_v = self.v.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        max_u.max(max_v)
    }

    /// Calculate velocity divergence at cell center (i, j)
    pub fn divergence(&self, mesh: &Mesh, i: usize, j: usize) -> f64 {
        let du_dx = (self.u[[i + 1, j]] - self.u[[i, j]]) / mesh.dx;
        let dv_dy = (self.v[[i, j + 1]] - self.v[[i, j]]) / mesh.dy;
        du_dx + dv_dy
    }

    /// Calculate total divergence (should be ~0 for incompressible flow)
    pub fn total_divergence(&self, mesh: &Mesh) -> f64 {
        let mut div_sum = 0.0;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                div_sum += self.divergence(mesh, i, j).abs();
            }
        }
        div_sum * mesh.dx * mesh.dy
    }

    /// Calculate total water volume (sum of VOF * cell volume)
    pub fn total_water_volume(&self, mesh: &Mesh) -> f64 {
        let cell_volume = mesh.dx * mesh.dy; // 2D: area
        self.vof.sum() * cell_volume
    }

    /// Get interpolated u-velocity at cell center (i, j)
    pub fn u_at_center(&self, i: usize, j: usize) -> f64 {
        0.5 * (self.u[[i, j]] + self.u[[i + 1, j]])
    }

    /// Get interpolated v-velocity at cell center (i, j)
    pub fn v_at_center(&self, i: usize, j: usize) -> f64 {
        0.5 * (self.v[[i, j]] + self.v[[i, j + 1]])
    }

    /// Get velocity magnitude at cell center
    pub fn velocity_magnitude(&self, i: usize, j: usize) -> f64 {
        let uc = self.u_at_center(i, j);
        let vc = self.v_at_center(i, j);
        (uc * uc + vc * vc).sqrt()
    }

    /// Clamp VOF values to [0, 1] range
    pub fn clamp_vof(&mut self) {
        for vof in self.vof.iter_mut() {
            *vof = vof.clamp(0.0, 1.0);
        }
    }

    /// Zero out velocity in air cells (VOF below threshold)
    /// This prevents spurious velocity diffusion in the air phase
    pub fn zero_air_velocity(&mut self, mesh: &Mesh, vof_threshold: f64) {
        // Zero u-velocity where adjacent cells are air
        for j in 0..mesh.ny {
            for i in 0..mesh.nx + 1 {
                let vof_face = if i == 0 {
                    self.vof[[0, j]]
                } else if i == mesh.nx {
                    self.vof[[mesh.nx - 1, j]]
                } else {
                    0.5 * (self.vof[[i - 1, j]] + self.vof[[i, j]])
                };
                
                if vof_face < vof_threshold {
                    self.u[[i, j]] = 0.0;
                }
            }
        }
        
        // Zero v-velocity where adjacent cells are air
        for j in 0..mesh.ny + 1 {
            for i in 0..mesh.nx {
                let vof_face = if j == 0 {
                    self.vof[[i, 0]]
                } else if j == mesh.ny {
                    self.vof[[i, mesh.ny - 1]]
                } else {
                    0.5 * (self.vof[[i, j - 1]] + self.vof[[i, j]])
                };
                
                if vof_face < vof_threshold {
                    self.v[[i, j]] = 0.0;
                }
            }
        }
    }

    /// Get density at u-velocity face (i, j) by averaging neighbors
    pub fn rho_at_u(&self, i: usize, j: usize, mesh: &Mesh) -> f64 {
        if i == 0 {
            self.rho[[0, j]]
        } else if i == mesh.nx {
            self.rho[[mesh.nx - 1, j]]
        } else {
            0.5 * (self.rho[[i - 1, j]] + self.rho[[i, j]])
        }
    }

    /// Get density at v-velocity face (i, j) by averaging neighbors
    pub fn rho_at_v(&self, i: usize, j: usize, mesh: &Mesh) -> f64 {
        if j == 0 {
            self.rho[[i, 0]]
        } else if j == mesh.ny {
            self.rho[[i, mesh.ny - 1]]
        } else {
            0.5 * (self.rho[[i, j - 1]] + self.rho[[i, j]])
        }
    }

    /// Find free surface elevation (eta) at column i from VOF field
    /// Returns y coordinate where VOF = 0.5 (interpolated)
    /// Skips solid cells (bathymetry)
    pub fn eta_at(&self, i: usize, mesh: &Mesh) -> f64 {
        // Scan from bottom to top to find water-air interface
        for j in 0..mesh.ny.saturating_sub(1) {
            // Skip solid cells
            if self.is_solid(i, j) || self.is_solid(i, j + 1) {
                continue;
            }
            
            let vof_below = self.vof[[i, j]];
            let vof_above = self.vof[[i, j + 1]];
            
            // Interface crosses between j and j+1 (water below, air above)
            if vof_below >= 0.5 && vof_above < 0.5 {
                let y_below = mesh.y_center(j);
                let y_above = mesh.y_center(j + 1);
                
                // Linear interpolation to find y where VOF = 0.5
                let t = (0.5 - vof_below) / (vof_above - vof_below);
                return y_below + t * (y_above - y_below);
            }
        }
        
        // Fallback: find highest fluid cell with VOF > 0.5
        for j in (0..mesh.ny).rev() {
            if !self.is_solid(i, j) && self.vof[[i, j]] >= 0.5 {
                return mesh.y_center(j) + 0.5 * mesh.dy;
            }
        }
        
        // No water found
        0.0
    }

    /// Find free surface elevation at x coordinate from VOF field
    /// Interpolates between columns
    pub fn eta_at_x(&self, x: f64, mesh: &Mesh) -> f64 {
        // Find column index
        let i_float = x / mesh.dx;
        let i = (i_float.floor() as usize).min(mesh.nx.saturating_sub(1));
        
        // If near boundary, just return single column value
        if i >= mesh.nx.saturating_sub(1) {
            return self.eta_at(i, mesh);
        }
        
        // Interpolate between columns
        let t = i_float - i as f64;
        let eta_left = self.eta_at(i, mesh);
        let eta_right = self.eta_at(i + 1, mesh);
        
        eta_left + t * (eta_right - eta_left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;

    fn test_mesh() -> Mesh {
        Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 0.5,
            nx: 10,
            ny: 5,
        })
    }

    #[test]
    fn test_field_dimensions() {
        let mesh = test_mesh();
        let fields = Fields::new(&mesh);
        
        assert_eq!(fields.vof.shape(), &[10, 5]);
        assert_eq!(fields.p.shape(), &[10, 5]);
        assert_eq!(fields.u.shape(), &[11, 5]);
        assert_eq!(fields.v.shape(), &[10, 6]);
    }

    #[test]
    fn test_initialize_water_level() {
        let mesh = test_mesh();
        let mut fields = Fields::new(&mesh);
        
        let ic = InitialConditionConfig { water_level: 0.25 };
        fields.initialize(&mesh, &ic, 1000.0, 1.225, 1e-6, 1.5e-5);
        
        // Bottom cells should be water (VOF = 1)
        assert!((fields.vof[[5, 0]] - 1.0).abs() < 0.1);
        
        // Top cells should be air (VOF = 0)
        assert!(fields.vof[[5, 4]] < 0.1);
        
        // Check density follows VOF
        assert!(fields.rho[[5, 0]] > 500.0); // Near water density
        assert!(fields.rho[[5, 4]] < 100.0); // Near air density
    }

    #[test]
    fn test_water_volume_conservation() {
        let mesh = test_mesh();
        let mut fields = Fields::new(&mesh);
        
        let ic = InitialConditionConfig { water_level: 0.25 };
        fields.initialize(&mesh, &ic, 1000.0, 1.225, 1e-6, 1.5e-5);
        
        let volume = fields.total_water_volume(&mesh);
        let expected = 1.0 * 0.25; // length_x * water_level
        
        assert!((volume - expected).abs() < 0.01);
    }
}

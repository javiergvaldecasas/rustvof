//! NetCDF-3 output for simulation results
//!
//! Writes CF-compliant NetCDF-3 files with all simulation fields.
//! Uses pure Rust implementation (no C dependencies).
//!
//! Note: NetCDF-3 files must be written in one pass, so all data is
//! accumulated in memory and written when `finalize()` is called.

use crate::mesh::Mesh;
use crate::fields::Fields;

/// NetCDF output writer
/// 
/// Accumulates timesteps in memory and writes to NetCDF-3 on finalize.
/// Can write multiple files when frames_per_file > 0.
pub struct NetcdfWriter {
    /// Output file prefix (without .nc extension)
    prefix: String,
    /// Frames per file (0 = single file)
    frames_per_file: usize,
    /// Current file index
    file_index: usize,
    /// Mesh dimensions
    nx: usize,
    ny: usize,
    /// Coordinate data
    x_coords: Vec<f32>,
    y_coords: Vec<f32>,
    /// Time values
    times: Vec<f64>,
    /// Field data (accumulated per timestep)
    vof_data: Vec<f32>,
    p_data: Vec<f32>,
    rho_data: Vec<f32>,
    u_data: Vec<f32>,
    v_data: Vec<f32>,
    eta_data: Vec<f32>,
    /// Turbulence fields (only if k-epsilon enabled)
    k_data: Vec<f32>,
    eps_data: Vec<f32>,
    nu_t_data: Vec<f32>,
    has_turbulence: bool,
    /// Solid mask (static, captured on first frame)
    solid_mask: Vec<i8>,
}

impl NetcdfWriter {
    /// Create a new NetCDF writer
    /// 
    /// # Arguments
    /// * `prefix` - Output file prefix (without .nc)
    /// * `frames_per_file` - Frames per file (0 = single file at end)
    pub fn new(prefix: &str, frames_per_file: usize) -> Self {
        Self {
            prefix: prefix.to_string(),
            frames_per_file,
            file_index: 0,
            nx: 0,
            ny: 0,
            x_coords: Vec::new(),
            y_coords: Vec::new(),
            times: Vec::new(),
            vof_data: Vec::new(),
            p_data: Vec::new(),
            rho_data: Vec::new(),
            u_data: Vec::new(),
            v_data: Vec::new(),
            eta_data: Vec::new(),
            k_data: Vec::new(),
            eps_data: Vec::new(),
            nu_t_data: Vec::new(),
            has_turbulence: false,
            solid_mask: Vec::new(),
        }
    }
    
    /// Get current output filepath
    fn current_filepath(&self) -> String {
        if self.frames_per_file == 0 {
            format!("{}.nc", self.prefix)
        } else {
            format!("{}_{:03}.nc", self.prefix, self.file_index)
        }
    }

    /// Buffer fields for the current timestep
    #[cfg(feature = "netcdf-output")]
    pub fn write(
        &mut self,
        mesh: &Mesh,
        fields: &Fields,
        time: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize coordinates and solid mask on first write
        if self.nx == 0 {
            self.nx = mesh.nx;
            self.ny = mesh.ny;
            self.x_coords = (0..mesh.nx).map(|i| mesh.x_center(i) as f32).collect();
            self.y_coords = (0..mesh.ny).map(|j| mesh.y_center(j) as f32).collect();
            
            // Capture solid mask (1 = solid, 0 = fluid)
            self.solid_mask = Vec::with_capacity(mesh.nx * mesh.ny);
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    self.solid_mask.push(if fields.is_solid(i, j) { 1 } else { 0 });
                }
            }
        }

        // Store time
        self.times.push(time);

        // Prepare field data in row-major order (y, x) for this timestep
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                self.vof_data.push(fields.vof[[i, j]] as f32);
                self.p_data.push(fields.p[[i, j]] as f32);
                self.rho_data.push(fields.rho[[i, j]] as f32);
                self.u_data.push(fields.u_at_center(i, j) as f32);
                self.v_data.push(fields.v_at_center(i, j) as f32);
            }
        }

        // Calculate and store eta
        let eta = self.calculate_eta(mesh, fields);
        self.eta_data.extend(eta);

        // Store turbulence fields if k-epsilon is active (k has non-zero values)
        let k_max: f64 = fields.k.iter().cloned().fold(0.0, f64::max);
        if k_max > 1e-10 {
            self.has_turbulence = true;
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    self.k_data.push(fields.k[[i, j]] as f32);
                    self.eps_data.push(fields.epsilon[[i, j]] as f32);
                    self.nu_t_data.push(fields.nu_t[[i, j]] as f32);
                }
            }
        }

        let frame_in_file = self.times.len();
        log::info!("Buffered NetCDF frame {} at t={:.4}s", frame_in_file, time);
        
        // Write file if we've reached the limit
        if self.frames_per_file > 0 && frame_in_file >= self.frames_per_file {
            self.finalize()?;
            self.file_index += 1;
        }
        
        Ok(())
    }

    /// Calculate free surface elevation (eta) from VOF field
    fn calculate_eta(&self, mesh: &Mesh, fields: &Fields) -> Vec<f32> {
        let mut eta = vec![-9999.0f32; mesh.nx];
        
        for i in 0..mesh.nx {
            for j in (1..mesh.ny).rev() {
                let vof_above = fields.vof[[i, j]];
                let vof_below = fields.vof[[i, j-1]];
                
                if (vof_above - 0.5) * (vof_below - 0.5) <= 0.0 {
                    let y_above = mesh.y_center(j);
                    let y_below = mesh.y_center(j-1);
                    
                    if (vof_above - vof_below).abs() > 1e-10 {
                        let t = (0.5 - vof_below) / (vof_above - vof_below);
                        eta[i] = (y_below + t * (y_above - y_below)) as f32;
                    } else {
                        eta[i] = ((y_above + y_below) / 2.0) as f32;
                    }
                    break;
                }
            }
            
            if eta[i] < -9000.0 {
                if fields.vof[[i, mesh.ny-1]] > 0.5 {
                    eta[i] = mesh.length_y as f32;
                } else if fields.vof[[i, 0]] < 0.5 {
                    eta[i] = 0.0;
                }
            }
        }
        eta
    }

    /// Write the accumulated data to the NetCDF file
    #[cfg(feature = "netcdf-output")]
    pub fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use netcdf3::{DataSet, FileWriter, Version};

        if self.times.is_empty() {
            return Ok(());
        }

        let nt = self.times.len();

        // Create dataset
        let mut ds = DataSet::new();

        // Helper macro to convert netcdf3 errors
        macro_rules! nc {
            ($expr:expr) => {
                $expr.map_err(|e| format!("NetCDF error: {:?}", e))?
            };
        }

        // Dimensions
        nc!(ds.set_unlimited_dim("time", nt));
        nc!(ds.add_fixed_dim("y", self.ny));
        nc!(ds.add_fixed_dim("x", self.nx));

        // Coordinate variables
        nc!(ds.add_var_f32("x", &["x"]));
        nc!(ds.add_var_attr_u8("x", "units", b"m".to_vec()));
        nc!(ds.add_var_attr_u8("x", "long_name", b"x-coordinate of cell center".to_vec()));

        nc!(ds.add_var_f32("y", &["y"]));
        nc!(ds.add_var_attr_u8("y", "units", b"m".to_vec()));
        nc!(ds.add_var_attr_u8("y", "long_name", b"y-coordinate of cell center".to_vec()));

        nc!(ds.add_var_f64("time", &["time"]));
        nc!(ds.add_var_attr_u8("time", "units", b"s".to_vec()));
        nc!(ds.add_var_attr_u8("time", "long_name", b"simulation time".to_vec()));

        // Field variables (time, y, x)
        nc!(ds.add_var_f32("vof", &["time", "y", "x"]));
        nc!(ds.add_var_attr_u8("vof", "units", b"1".to_vec()));
        nc!(ds.add_var_attr_u8("vof", "long_name", b"volume of fluid fraction".to_vec()));

        nc!(ds.add_var_f32("p", &["time", "y", "x"]));
        nc!(ds.add_var_attr_u8("p", "units", b"Pa".to_vec()));
        nc!(ds.add_var_attr_u8("p", "long_name", b"pressure".to_vec()));

        nc!(ds.add_var_f32("rho", &["time", "y", "x"]));
        nc!(ds.add_var_attr_u8("rho", "units", b"kg m-3".to_vec()));
        nc!(ds.add_var_attr_u8("rho", "long_name", b"density".to_vec()));

        nc!(ds.add_var_f32("u", &["time", "y", "x"]));
        nc!(ds.add_var_attr_u8("u", "units", b"m s-1".to_vec()));
        nc!(ds.add_var_attr_u8("u", "long_name", b"x-velocity at cell center".to_vec()));

        nc!(ds.add_var_f32("v", &["time", "y", "x"]));
        nc!(ds.add_var_attr_u8("v", "units", b"m s-1".to_vec()));
        nc!(ds.add_var_attr_u8("v", "long_name", b"y-velocity at cell center".to_vec()));

        nc!(ds.add_var_f32("eta", &["time", "x"]));
        nc!(ds.add_var_attr_u8("eta", "units", b"m".to_vec()));
        nc!(ds.add_var_attr_u8("eta", "long_name", b"free surface elevation".to_vec()));

        // Solid mask (static field, no time dimension)
        nc!(ds.add_var_i8("solid", &["y", "x"]));
        nc!(ds.add_var_attr_u8("solid", "units", b"1".to_vec()));
        nc!(ds.add_var_attr_u8("solid", "long_name", b"solid mask (1=solid, 0=fluid)".to_vec()));

        // Turbulence fields (only if k-epsilon was active)
        if self.has_turbulence {
            nc!(ds.add_var_f32("k", &["time", "y", "x"]));
            nc!(ds.add_var_attr_u8("k", "units", b"m2 s-2".to_vec()));
            nc!(ds.add_var_attr_u8("k", "long_name", b"turbulent kinetic energy".to_vec()));

            nc!(ds.add_var_f32("epsilon", &["time", "y", "x"]));
            nc!(ds.add_var_attr_u8("epsilon", "units", b"m2 s-3".to_vec()));
            nc!(ds.add_var_attr_u8("epsilon", "long_name", b"turbulent dissipation rate".to_vec()));

            nc!(ds.add_var_f32("nu_t", &["time", "y", "x"]));
            nc!(ds.add_var_attr_u8("nu_t", "units", b"m2 s-1".to_vec()));
            nc!(ds.add_var_attr_u8("nu_t", "long_name", b"turbulent viscosity".to_vec()));
        }

        // Global attributes
        nc!(ds.add_global_attr_u8("title", b"VOF2D simulation output".to_vec()));
        nc!(ds.add_global_attr_u8("institution", b"VOF2D Solver".to_vec()));
        nc!(ds.add_global_attr_u8("Conventions", b"CF-1.8".to_vec()));

        // Write file (remove existing file first)
        let filepath = self.current_filepath();
        if std::path::Path::new(&filepath).exists() {
            std::fs::remove_file(&filepath)
                .map_err(|e| format!("Failed to remove existing file {}: {}", filepath, e))?;
        }
        let mut writer = nc!(FileWriter::create_new(&filepath));
        nc!(writer.set_def(&ds, Version::Classic, 0));

        // Write coordinate data
        nc!(writer.write_var_f32("x", &self.x_coords));
        nc!(writer.write_var_f32("y", &self.y_coords));
        nc!(writer.write_var_f64("time", &self.times));

        // Write field data
        nc!(writer.write_var_f32("vof", &self.vof_data));
        nc!(writer.write_var_f32("p", &self.p_data));
        nc!(writer.write_var_f32("rho", &self.rho_data));
        nc!(writer.write_var_f32("u", &self.u_data));
        nc!(writer.write_var_f32("v", &self.v_data));
        nc!(writer.write_var_f32("eta", &self.eta_data));
        nc!(writer.write_var_i8("solid", &self.solid_mask));

        // Write turbulence fields if present
        if self.has_turbulence {
            nc!(writer.write_var_f32("k", &self.k_data));
            nc!(writer.write_var_f32("epsilon", &self.eps_data));
            nc!(writer.write_var_f32("nu_t", &self.nu_t_data));
        }

        nc!(writer.close());

        log::info!("Wrote NetCDF file: {} ({} timesteps, {}x{} grid)", 
            filepath, nt, self.nx, self.ny);

        // Clear buffers
        self.times.clear();
        self.vof_data.clear();
        self.p_data.clear();
        self.rho_data.clear();
        self.u_data.clear();
        self.v_data.clear();
        self.eta_data.clear();
        self.k_data.clear();
        self.eps_data.clear();
        self.nu_t_data.clear();
        // Note: has_turbulence is kept across files

        Ok(())
    }

    /// Get current frame number
    pub fn frame(&self) -> usize {
        self.times.len()
    }

    /// Get output prefix
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Stub for when netcdf feature is disabled
    #[cfg(not(feature = "netcdf-output"))]
    pub fn write(
        &mut self,
        _mesh: &Mesh,
        _fields: &Fields,
        _time: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err("NetCDF output requires the 'netcdf-output' feature. \
             Rebuild with: cargo build --features netcdf-output".into())
    }

    #[cfg(not(feature = "netcdf-output"))]
    pub fn finalize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[cfg(feature = "netcdf-output")]
impl Drop for NetcdfWriter {
    fn drop(&mut self) {
        if !self.times.is_empty() {
            if let Err(e) = self.finalize() {
                log::error!("Error finalizing NetCDF file: {}", e);
            }
        }
    }
}

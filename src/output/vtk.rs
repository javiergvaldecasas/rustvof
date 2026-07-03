//! VTK file output for visualization in ParaView
//!
//! Uses VTK Legacy format (RectilinearGrid) for simplicity

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// VTK output writer
pub struct VtkWriter {
    /// Output directory
    pub output_dir: String,
    /// File prefix
    pub prefix: String,
    /// Frame counter
    frame: usize,
}

impl VtkWriter {
    /// Create a new VTK writer
    pub fn new(prefix: &str) -> Self {
        // Extract directory from prefix
        let path = Path::new(prefix);
        let output_dir = path.parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let file_prefix = path.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        
        // Create output directory
        if !output_dir.is_empty() && output_dir != "." {
            fs::create_dir_all(&output_dir).ok();
        }
        
        Self {
            output_dir,
            prefix: file_prefix,
            frame: 0,
        }
    }

    /// Write fields to VTK file
    pub fn write(
        &mut self,
        mesh: &Mesh,
        fields: &Fields,
        time: f64,
    ) -> std::io::Result<()> {
        let filename = format!(
            "{}/{}_{:06}.vtk",
            self.output_dir,
            self.prefix,
            self.frame
        );
        
        let file = File::create(&filename)?;
        let mut writer = BufWriter::new(file);
        
        // VTK Header
        writeln!(writer, "# vtk DataFile Version 3.0")?;
        writeln!(writer, "VOF2D output t={:.6}", time)?;
        writeln!(writer, "ASCII")?;
        writeln!(writer, "DATASET RECTILINEAR_GRID")?;
        writeln!(writer, "DIMENSIONS {} {} 1", mesh.nx + 1, mesh.ny + 1)?;
        
        // X coordinates (cell corners)
        writeln!(writer, "X_COORDINATES {} float", mesh.nx + 1)?;
        for i in 0..=mesh.nx {
            write!(writer, "{:.6} ", i as f64 * mesh.dx)?;
        }
        writeln!(writer)?;
        
        // Y coordinates
        writeln!(writer, "Y_COORDINATES {} float", mesh.ny + 1)?;
        for j in 0..=mesh.ny {
            write!(writer, "{:.6} ", j as f64 * mesh.dy)?;
        }
        writeln!(writer)?;
        
        // Z coordinates (2D, so just 0)
        writeln!(writer, "Z_COORDINATES 1 float")?;
        writeln!(writer, "0.0")?;
        
        // Cell data
        writeln!(writer, "CELL_DATA {}", mesh.nx * mesh.ny)?;
        
        // VOF field
        writeln!(writer, "SCALARS vof float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                writeln!(writer, "{:.6}", fields.vof[[i, j]])?;
            }
        }
        
        // Pressure field
        writeln!(writer, "SCALARS pressure float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                writeln!(writer, "{:.6}", fields.p[[i, j]])?;
            }
        }
        
        // Density field
        writeln!(writer, "SCALARS density float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                writeln!(writer, "{:.6}", fields.rho[[i, j]])?;
            }
        }
        
        // Velocity field (interpolated to cell centers)
        writeln!(writer, "VECTORS velocity float")?;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                let u = fields.u_at_center(i, j);
                let v = fields.v_at_center(i, j);
                writeln!(writer, "{:.6} {:.6} 0.0", u, v)?;
            }
        }
        
        // Velocity magnitude
        writeln!(writer, "SCALARS velocity_magnitude float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                writeln!(writer, "{:.6}", fields.velocity_magnitude(i, j))?;
            }
        }
        
        // Turbulent kinetic energy (k) - only if non-zero
        let k_max = fields.k.iter().cloned().fold(0.0_f64, f64::max);
        if k_max > 1e-10 {
            writeln!(writer, "SCALARS k float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    writeln!(writer, "{:.6e}", fields.k[[i, j]])?;
                }
            }
            
            // Turbulent dissipation rate (epsilon)
            writeln!(writer, "SCALARS epsilon float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    writeln!(writer, "{:.6e}", fields.epsilon[[i, j]])?;
                }
            }
            
            // Turbulent viscosity (nu_t)
            writeln!(writer, "SCALARS nu_t float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..mesh.ny {
                for i in 0..mesh.nx {
                    writeln!(writer, "{:.6e}", fields.nu_t[[i, j]])?;
                }
            }
        }
        
        self.frame += 1;
        
        log::info!("Wrote VTK frame {} at t={:.4}s: {}", self.frame, time, filename);
        
        Ok(())
    }

    /// Get current frame number
    pub fn frame(&self) -> usize {
        self.frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;
    use std::path::PathBuf;

    #[test]
    fn test_vtk_writer() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 0.5,
            nx: 10,
            ny: 5,
        });
        
        let fields = Fields::new(&mesh);
        
        // Use temp directory
        let temp_dir = std::env::temp_dir();
        let prefix = temp_dir.join("vof2d_test/output");
        
        let mut writer = VtkWriter::new(prefix.to_str().unwrap());
        
        let result = writer.write(&mesh, &fields, 0.0);
        assert!(result.is_ok());
        
        // Check file exists
        let expected_file = temp_dir.join("vof2d_test/output_000000.vtk");
        assert!(expected_file.exists());
        
        // Cleanup
        fs::remove_file(expected_file).ok();
        fs::remove_dir(temp_dir.join("vof2d_test")).ok();
    }
}

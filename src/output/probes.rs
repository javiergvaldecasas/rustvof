//! Probe output for time series data
//!
//! Exports free surface elevation at wave gauge locations.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::config::ProbesConfig;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// Probe manager for time series output
pub struct ProbeWriter {
    /// Configuration
    config: ProbesConfig,
    /// Output file writer
    writer: Option<BufWriter<File>>,
    /// Last output time
    last_output_time: f64,
    /// Column indices for wave gauges (precomputed)
    gauge_columns: Vec<usize>,
}

impl ProbeWriter {
    /// Create a new probe writer
    pub fn new(config: &ProbesConfig, mesh: &Mesh, output_prefix: &str) -> Self {
        if !config.enabled || config.wave_gauges.is_empty() {
            return Self {
                config: config.clone(),
                writer: None,
                last_output_time: -1.0,
                gauge_columns: vec![],
            };
        }
        
        // Precompute column indices for each wave gauge
        let gauge_columns: Vec<usize> = config.wave_gauges.iter()
            .map(|&x| {
                // Find closest cell column
                let i = ((x / mesh.dx) as usize).min(mesh.nx - 1);
                i
            })
            .collect();
        
        // Create output file
        let file_path = if config.file.starts_with('/') {
            config.file.clone()
        } else {
            // Relative to output prefix directory
            let prefix_path = Path::new(output_prefix);
            let dir = prefix_path.parent().unwrap_or(Path::new("."));
            dir.join(&config.file).to_string_lossy().to_string()
        };
        
        let file = match File::create(&file_path) {
            Ok(f) => f,
            Err(e) => {
                log::error!("Failed to create probe file {}: {}", file_path, e);
                return Self {
                    config: config.clone(),
                    writer: None,
                    last_output_time: -1.0,
                    gauge_columns,
                };
            }
        };
        
        let mut writer = BufWriter::new(file);
        
        // Write header
        let mut header = String::from("time");
        for (i, &x) in config.wave_gauges.iter().enumerate() {
            header.push_str(&format!(",wg{}_{:.3}m", i + 1, x));
        }
        header.push('\n');
        
        if let Err(e) = writer.write_all(header.as_bytes()) {
            log::error!("Failed to write probe header: {}", e);
        }
        
        log::info!("Probe output enabled: {} wave gauges -> {}", 
            config.wave_gauges.len(), file_path);
        
        Self {
            config: config.clone(),
            writer: Some(writer),
            last_output_time: -1.0,
            gauge_columns,
        }
    }
    
    /// Check if probes are enabled
    pub fn is_enabled(&self) -> bool {
        self.writer.is_some()
    }
    
    /// Record probe data at current time
    /// Returns true if data was written
    pub fn record(&mut self, time: f64, mesh: &Mesh, fields: &Fields, dt_output: f64) -> bool {
        let writer = match &mut self.writer {
            Some(w) => w,
            None => return false,
        };
        
        // Check if we should output
        let dt = self.config.dt.unwrap_or(dt_output);
        if time - self.last_output_time < dt * 0.99 && self.last_output_time >= 0.0 {
            return false;
        }
        
        // Compute free surface at each gauge
        let mut line = format!("{:.6}", time);
        
        for &col in &self.gauge_columns {
            let eta = Self::find_free_surface(col, mesh, fields);
            line.push_str(&format!(",{:.6}", eta));
        }
        line.push('\n');
        
        if let Err(e) = writer.write_all(line.as_bytes()) {
            log::error!("Failed to write probe data: {}", e);
            return false;
        }
        
        // Flush periodically to ensure data is written even if process is killed
        if let Err(e) = writer.flush() {
            log::error!("Failed to flush probe data: {}", e);
        }
        
        self.last_output_time = time;
        true
    }
    
    /// Find free surface elevation at column i
    /// Returns y coordinate where VOF = 0.5 (interpolated)
    /// Properly handles bathymetry by skipping solid cells
    fn find_free_surface(i: usize, mesh: &Mesh, fields: &Fields) -> f64 {
        // Scan from bottom to top to find the water-air interface
        // Skip solid cells (bathymetry) - they don't contain water
        for j in 0..mesh.ny - 1 {
            // Skip if current cell is solid
            if fields.is_solid(i, j) {
                continue;
            }
            
            // Skip if next cell is solid (can't interpolate across solid boundary)
            if fields.is_solid(i, j + 1) {
                continue;
            }
            
            let vof_below = fields.vof[[i, j]];
            let vof_above = fields.vof[[i, j + 1]];
            
            // Interface crosses between j and j+1 (water below, air above)
            if vof_below >= 0.5 && vof_above < 0.5 {
                // Linear interpolation to find exact crossing
                let y_below = (j as f64 + 0.5) * mesh.dy;
                let y_above = (j as f64 + 1.5) * mesh.dy;
                
                // Interpolate: find y where VOF = 0.5
                let t = (0.5 - vof_below) / (vof_above - vof_below);
                let eta = y_below + t * (y_above - y_below);
                return eta;
            }
        }
        
        // Fallback: find highest fluid cell with VOF > 0.5
        for j in (0..mesh.ny).rev() {
            if !fields.is_solid(i, j) && fields.vof[[i, j]] >= 0.5 {
                return (j as f64 + 1.0) * mesh.dy;
            }
        }
        
        // No water found in fluid cells - return 0 (dry)
        0.0
    }
    
    /// Flush buffer to disk
    pub fn flush(&mut self) {
        if let Some(ref mut writer) = self.writer {
            let _ = writer.flush();
        }
    }
}

impl Drop for ProbeWriter {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;
    
    #[test]
    fn test_find_free_surface() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut fields = Fields::new(&mesh);
        
        // Set water level at y = 0.5 (VOF = 1 below, 0 above)
        for j in 0..mesh.ny {
            let y = (j as f64 + 0.5) * mesh.dy;
            for i in 0..mesh.nx {
                fields.vof[[i, j]] = if y < 0.5 { 1.0 } else { 0.0 };
            }
        }
        
        let eta = ProbeWriter::find_free_surface(5, &mesh, &fields);
        assert!((eta - 0.5).abs() < mesh.dy, "Expected eta ≈ 0.5, got {}", eta);
    }
}

//! Fluid properties management

use crate::config::FluidConfig;

/// Fluid properties container
#[derive(Debug, Clone, Copy)]
pub struct FluidProperties {
    /// Water density (kg/m³)
    pub rho_water: f64,
    /// Air density (kg/m³)
    pub rho_air: f64,
    /// Water kinematic viscosity (m²/s)
    pub nu_water: f64,
    /// Air kinematic viscosity (m²/s)
    pub nu_air: f64,
    /// Gravitational acceleration (m/s²)
    pub gravity: f64,
}

impl FluidProperties {
    /// Create from configuration
    pub fn from_config(config: &FluidConfig) -> Self {
        Self {
            rho_water: config.rho_water,
            rho_air: config.rho_air,
            nu_water: config.nu_water,
            nu_air: config.nu_air,
            gravity: config.gravity,
        }
    }

    /// Interpolate density based on VOF
    pub fn density(&self, vof: f64) -> f64 {
        vof * self.rho_water + (1.0 - vof) * self.rho_air
    }

    /// Interpolate kinematic viscosity based on VOF
    pub fn viscosity(&self, vof: f64) -> f64 {
        vof * self.nu_water + (1.0 - vof) * self.nu_air
    }

    /// Interpolate dynamic viscosity based on VOF
    pub fn dynamic_viscosity(&self, vof: f64) -> f64 {
        self.density(vof) * self.viscosity(vof)
    }

    /// Density ratio (water/air)
    pub fn density_ratio(&self) -> f64 {
        self.rho_water / self.rho_air
    }

    /// Viscosity ratio (water/air)
    pub fn viscosity_ratio(&self) -> f64 {
        self.nu_water / self.nu_air
    }
}

impl Default for FluidProperties {
    fn default() -> Self {
        Self {
            rho_water: 1000.0,
            rho_air: 1.225,
            nu_water: 1.0e-6,
            nu_air: 1.5e-5,
            gravity: 9.81,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_interpolation() {
        let props = FluidProperties::default();
        
        // Pure water
        assert!((props.density(1.0) - 1000.0).abs() < 1e-10);
        
        // Pure air
        assert!((props.density(0.0) - 1.225).abs() < 1e-10);
        
        // 50-50 mix
        let expected = 0.5 * 1000.0 + 0.5 * 1.225;
        assert!((props.density(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_viscosity_interpolation() {
        let props = FluidProperties::default();
        
        // Pure water
        assert!((props.viscosity(1.0) - 1.0e-6).abs() < 1e-15);
        
        // Pure air
        assert!((props.viscosity(0.0) - 1.5e-5).abs() < 1e-15);
    }
}

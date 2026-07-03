//! Cell type classification
//!
//! Each cell can be classified as fluid (water/air determined by VOF),
//! solid (blocked), or porous (partial permeability).

/// Cell type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CellType {
    /// Fluid cell - VOF determines water/air fraction
    #[default]
    Fluid = 0,
    
    /// Solid cell - completely blocked, no flow (permanent: bathymetry, walls)
    /// Velocity = 0, excluded from pressure solve
    Solid = 1,
    
    /// Porous cell - partial permeability (Darcy/Forchheimer)
    /// For future: breakwaters, rubble mound, vegetation
    Porous = 2,
    
    /// Paddle cell - temporary solid behind wave paddle (resets as paddle moves)
    /// Behaves like Solid but can be converted back to Fluid
    Paddle = 3,
}

impl CellType {
    /// Check if cell is fluid (water or air)
    #[inline]
    pub fn is_fluid(self) -> bool {
        self == CellType::Fluid
    }
    
    /// Check if cell is solid (completely blocked) - includes Solid and Paddle
    #[inline]
    pub fn is_solid(self) -> bool {
        self == CellType::Solid || self == CellType::Paddle
    }
    
    /// Check if cell is permanent solid (bathymetry, walls - NOT paddle)
    #[inline]
    pub fn is_permanent_solid(self) -> bool {
        self == CellType::Solid
    }
    
    /// Check if cell is paddle (temporary solid)
    #[inline]
    pub fn is_paddle(self) -> bool {
        self == CellType::Paddle
    }
    
    /// Check if cell is porous
    #[inline]
    pub fn is_porous(self) -> bool {
        self == CellType::Porous
    }
    
    /// Check if flow is allowed (fluid or porous)
    #[inline]
    pub fn allows_flow(self) -> bool {
        self == CellType::Fluid || self == CellType::Porous
    }
    
    /// Convert from u8 for serialization
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => CellType::Fluid,
            1 => CellType::Solid,
            2 => CellType::Porous,
            3 => CellType::Paddle,
            _ => CellType::Fluid,
        }
    }
}

/// Porous media properties (for future use)
#[derive(Debug, Clone, Copy)]
pub struct PorousProperties {
    /// Porosity n (0-1), fraction of void space
    pub porosity: f64,
    
    /// Linear drag coefficient α (Darcy term)
    /// Resistance = α * u
    pub alpha: f64,
    
    /// Quadratic drag coefficient β (Forchheimer term)
    /// Resistance = β * |u| * u
    pub beta: f64,
    
    /// Added mass coefficient C_m
    pub added_mass: f64,
}

impl Default for PorousProperties {
    fn default() -> Self {
        Self {
            porosity: 0.4,      // Typical for rubble mound
            alpha: 200.0,       // Linear resistance
            beta: 1.1,          // Quadratic resistance
            added_mass: 0.34,   // Added mass coefficient
        }
    }
}

impl PorousProperties {
    /// Create properties for rubble mound (typical breakwater core)
    pub fn rubble_mound() -> Self {
        Self {
            porosity: 0.4,
            alpha: 200.0,
            beta: 1.1,
            added_mass: 0.34,
        }
    }
    
    /// Create properties for armor layer (larger stones)
    pub fn armor_layer() -> Self {
        Self {
            porosity: 0.5,
            alpha: 100.0,
            beta: 0.8,
            added_mass: 0.34,
        }
    }
    
    /// Create properties for vegetation
    pub fn vegetation(density: f64) -> Self {
        Self {
            porosity: 1.0 - density * 0.01,  // Approximate
            alpha: 50.0 * density,
            beta: 0.5 * density,
            added_mass: 0.0,
        }
    }
    
    /// Compute Darcy-Forchheimer resistance force
    /// F = -(α + β|u|) * u / n²
    pub fn resistance(&self, u: f64, v: f64) -> (f64, f64) {
        let speed = (u * u + v * v).sqrt();
        let n2 = self.porosity * self.porosity;
        let coeff = (self.alpha + self.beta * speed) / n2;
        (-coeff * u, -coeff * v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cell_type_default() {
        let ct: CellType = Default::default();
        assert_eq!(ct, CellType::Fluid);
    }
    
    #[test]
    fn test_cell_type_checks() {
        assert!(CellType::Fluid.is_fluid());
        assert!(CellType::Solid.is_solid());
        assert!(CellType::Porous.is_porous());
        
        assert!(CellType::Fluid.allows_flow());
        assert!(!CellType::Solid.allows_flow());
        assert!(CellType::Porous.allows_flow());
    }
    
    #[test]
    fn test_porous_resistance() {
        let props = PorousProperties::default();
        let (fx, fy) = props.resistance(1.0, 0.0);
        
        // Resistance should oppose flow
        assert!(fx < 0.0);
        assert!(fy.abs() < 1e-10);
    }
}

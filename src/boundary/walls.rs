//! Solid wall boundary conditions
//!
//! Implements no-slip (u=0) and free-slip (∂u/∂n=0) conditions

use crate::mesh::Mesh;
use crate::fields::Fields;

/// Wall boundary condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallType {
    /// No-slip: velocity = 0 at wall
    NoSlip,
    /// Free-slip: normal velocity = 0, tangential free
    FreeSlip,
}

/// Apply wall boundary conditions to velocity field
/// 
/// - Left/Right walls: u = 0, ∂v/∂x = 0
/// - Bottom wall: v = 0, ∂u/∂y = 0
/// - Top: free surface (handled separately)
pub fn apply_wall_bc(
    fields: &mut Fields,
    mesh: &Mesh,
    wall_type: WallType,
) {
    match wall_type {
        WallType::NoSlip => apply_no_slip(fields, mesh),
        WallType::FreeSlip => apply_free_slip(fields, mesh),
    }
}

/// Apply no-slip boundary conditions
fn apply_no_slip(fields: &mut Fields, mesh: &Mesh) {
    // Left wall (x = 0): u = 0
    for j in 0..mesh.ny {
        fields.u[[0, j]] = 0.0;
    }
    
    // Right wall (x = L): u = 0
    for j in 0..mesh.ny {
        fields.u[[mesh.nx, j]] = 0.0;
    }
    
    // Bottom wall (y = 0): v = 0
    for i in 0..mesh.nx {
        fields.v[[i, 0]] = 0.0;
    }
    
    // Top boundary (y = H): free surface - v extrapolated or set based on physics
    // For closed tank: v = 0
    // For open: leave as is or set to zero gradient
    for i in 0..mesh.nx {
        fields.v[[i, mesh.ny]] = 0.0;
    }
}

/// Apply free-slip boundary conditions
fn apply_free_slip(fields: &mut Fields, mesh: &Mesh) {
    // Left wall: u = 0, ∂v/∂x = 0 (copy from interior)
    for j in 0..mesh.ny {
        fields.u[[0, j]] = 0.0;
    }
    for j in 0..mesh.ny + 1 {
        // v at left boundary = v at first interior
        fields.v[[0, j]] = if mesh.nx > 1 { fields.v[[1, j]] } else { 0.0 };
    }
    
    // Right wall: u = 0, ∂v/∂x = 0
    for j in 0..mesh.ny {
        fields.u[[mesh.nx, j]] = 0.0;
    }
    for j in 0..mesh.ny + 1 {
        let last = mesh.nx - 1;
        fields.v[[last, j]] = if mesh.nx > 1 { fields.v[[last - 1, j]] } else { 0.0 };
    }
    
    // Bottom wall: v = 0, ∂u/∂y = 0
    for i in 0..mesh.nx {
        fields.v[[i, 0]] = 0.0;
    }
    for i in 0..mesh.nx + 1 {
        fields.u[[i, 0]] = if mesh.ny > 1 { fields.u[[i, 1]] } else { 0.0 };
    }
    
    // Top: free surface
    for i in 0..mesh.nx {
        fields.v[[i, mesh.ny]] = 0.0;
    }
    for i in 0..mesh.nx + 1 {
        let last = mesh.ny - 1;
        fields.u[[i, last]] = if mesh.ny > 1 { fields.u[[i, last - 1]] } else { 0.0 };
    }
}

/// Apply boundary condition for pressure (Neumann: ∂p/∂n = 0)
/// This is implicitly handled in the pressure solver
#[allow(dead_code)]
pub fn apply_pressure_bc(_fields: &mut Fields, _mesh: &Mesh) {
    // Copy pressure to ghost cells (if needed)
    // For now, handled directly in pressure solver
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;

    #[test]
    fn test_no_slip_bc() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut fields = Fields::new(&mesh);
        
        // Set some non-zero velocities
        for j in 0..mesh.ny {
            for i in 0..mesh.nx + 1 {
                fields.u[[i, j]] = 1.0;
            }
        }
        
        apply_wall_bc(&mut fields, &mesh, WallType::NoSlip);
        
        // Check left wall
        for j in 0..mesh.ny {
            assert!((fields.u[[0, j]]).abs() < 1e-10);
        }
        
        // Check right wall
        for j in 0..mesh.ny {
            assert!((fields.u[[mesh.nx, j]]).abs() < 1e-10);
        }
        
        // Check bottom wall
        for i in 0..mesh.nx {
            assert!((fields.v[[i, 0]]).abs() < 1e-10);
        }
    }
}

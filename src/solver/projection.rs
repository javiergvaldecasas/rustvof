//! Velocity projection step
//!
//! Corrects intermediate velocity to be divergence-free:
//! u^{n+1} = u* - (Δt/ρ) ∇p

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// Project velocity to be divergence-free
/// 
/// u^{n+1} = u* - Δt * ∂p/∂x
/// v^{n+1} = v* - Δt * ∂p/∂y
/// 
/// Note: For variable density, this should be u* - (Δt/ρ) ∇p
/// For simplicity in MVP, we use constant effective density
pub fn project_velocity(
    fields: &mut Fields,
    mesh: &Mesh,
    dt: f64,
) {
    // Correct u-velocity (on vertical faces)
    for j in 0..mesh.ny {
        for i in 1..mesh.nx {  // Interior faces only
            let dp_dx = (fields.p[[i, j]] - fields.p[[i - 1, j]]) / mesh.dx;
            fields.u[[i, j]] -= dt * dp_dx;
        }
    }
    
    // Correct v-velocity (on horizontal faces)
    for j in 1..mesh.ny {  // Interior faces only
        for i in 0..mesh.nx {
            let dp_dy = (fields.p[[i, j]] - fields.p[[i, j - 1]]) / mesh.dy;
            fields.v[[i, j]] -= dt * dp_dy;
        }
    }
}

/// Project velocity with variable density
/// Uses cell_type to skip solid cells
pub fn project_velocity_variable_density(
    fields: &mut Fields,
    mesh: &Mesh,
    dt: f64,
) {
    // Correct u-velocity
    for j in 0..mesh.ny {
        for i in 1..mesh.nx {
            // Skip if either neighboring cell is solid
            let left_solid = fields.is_solid(i - 1, j);
            let right_solid = fields.is_solid(i, j);
            if left_solid || right_solid {
                fields.u[[i, j]] = 0.0;
                continue;
            }
            
            let dp_dx = (fields.p[[i, j]] - fields.p[[i - 1, j]]) / mesh.dx;
            let rho_face = fields.rho_at_u(i, j, mesh);
            fields.u[[i, j]] -= dt * dp_dx / rho_face;
        }
    }
    
    // Correct v-velocity
    for j in 1..mesh.ny {
        for i in 0..mesh.nx {
            // Skip if either neighboring cell is solid
            let below_solid = fields.is_solid(i, j - 1);
            let above_solid = fields.is_solid(i, j);
            if below_solid || above_solid {
                fields.v[[i, j]] = 0.0;
                continue;
            }
            
            let dp_dy = (fields.p[[i, j]] - fields.p[[i, j - 1]]) / mesh.dy;
            let rho_face = fields.rho_at_v(i, j, mesh);
            fields.v[[i, j]] -= dt * dp_dy / rho_face;
        }
    }
}

/// Compute intermediate velocity (before pressure correction)
///
/// u* = u^n + Δt * (-advection + diffusion + gravity)
///
/// Gravity is applied to the vertical velocity weighted by the VOF fraction
/// interpolated onto the v-face (g_effective = gravity * vof_face), so gravity
/// acts on water (F ≈ 1) but not on air (F ≈ 0).
///
/// Note: this simple VOF-weighted term can introduce small artificial gradients
/// at the interface (the pressure gradient does not exactly balance the
/// discontinuous gravity term), which may generate weak spurious velocities near
/// the free surface. A reduced-density buoyancy form, g * (ρ - ρ_air) / ρ, would
/// mitigate this but is not the path currently used here.
pub fn compute_intermediate_velocity(
    fields: &mut Fields,
    adv_u: &Array2<f64>,
    adv_v: &Array2<f64>,
    diff_u: &Array2<f64>,
    diff_v: &Array2<f64>,
    mesh: &Mesh,
    dt: f64,
    gravity: f64,
) {
    let (nu_x, nu_y) = mesh.num_u();
    let (nv_x, nv_y) = mesh.num_v();
    
    // Update u-velocity (no gravity in x)
    for j in 0..nu_y {
        for i in 1..nu_x - 1 {  // Interior only
            // Skip if neighboring cells are solid
            let left_solid = i > 0 && fields.is_solid(i - 1, j);
            let right_solid = i < mesh.nx && fields.is_solid(i, j);
            if left_solid || right_solid {
                fields.u[[i, j]] = 0.0;
                continue;
            }
            fields.u[[i, j]] += dt * (-adv_u[[i, j]] + diff_u[[i, j]]);
        }
    }
    
    // Update v-velocity with gravity (points downward = negative)
    for j in 1..nv_y - 1 {  // Interior only
        for i in 0..nv_x {
            // Check for solid neighbors
            let below_solid = j > 0 && fields.is_solid(i, j - 1);
            let above_solid = j < mesh.ny && fields.is_solid(i, j);
            
            // If face touches solid, zero velocity
            if below_solid || above_solid {
                fields.v[[i, j]] = 0.0;
                continue;
            }
            
            // Get VOF at this v-face
            let vof_below = if j > 0 { fields.vof[[i, j - 1]] } else { 0.0 };
            let vof_above = if j < mesh.ny { fields.vof[[i, j]] } else { 0.0 };
            let vof_face = 0.5 * (vof_below + vof_above);
            
            // Apply gravity weighted by VOF
            let g_effective = gravity * vof_face;
            
            fields.v[[i, j]] += dt * (-adv_v[[i, j]] + diff_v[[i, j]] - g_effective);
        }
    }
}

/// Enforce no-penetration boundary condition on solid interfaces
/// Call this AFTER projection to ensure v=0 at solid boundaries
pub fn enforce_solid_bc(fields: &mut Fields, mesh: &Mesh) {
    // Zero v-velocity at faces touching solid cells
    for j in 0..mesh.ny + 1 {
        for i in 0..mesh.nx {
            let below_solid = j > 0 && j <= mesh.ny && fields.is_solid(i, j - 1);
            let above_solid = j < mesh.ny && fields.is_solid(i, j);
            
            if below_solid || above_solid {
                fields.v[[i, j]] = 0.0;
            }
        }
    }
    
    // Zero u-velocity at faces touching solid cells
    for j in 0..mesh.ny {
        for i in 0..mesh.nx + 1 {
            let left_solid = i > 0 && fields.is_solid(i - 1, j);
            let right_solid = i < mesh.nx && fields.is_solid(i, j);
            
            if left_solid || right_solid {
                fields.u[[i, j]] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;

    #[test]
    fn test_projection_modifies_velocity() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut fields = Fields::new(&mesh);
        
        // Set up a uniform velocity field
        for j in 0..mesh.ny {
            for i in 0..mesh.nx + 1 {
                fields.u[[i, j]] = 1.0;
            }
        }
        
        // Set up a pressure gradient
        for j in 0..mesh.ny {
            for i in 0..mesh.nx {
                fields.p[[i, j]] = i as f64 * 0.1;
            }
        }
        
        let u_before = fields.u[[5, 5]];
        
        project_velocity(&mut fields, &mesh, 1.0);
        
        let u_after = fields.u[[5, 5]];
        
        // Velocity should be modified by pressure gradient
        assert!((u_after - u_before).abs() > 1e-10);
    }
}

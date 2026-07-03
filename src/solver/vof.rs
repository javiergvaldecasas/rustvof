//! VOF (Volume of Fluid) advection
//!
//! Solves: ∂F/∂t + ∇·(Fu) = 0
//! Where F ∈ [0,1] is the volume fraction (0=air, 1=water)
//!
//! Uses TVD scheme with Van Leer limiter for reduced numerical diffusion.

use crate::mesh::Mesh;
use crate::fields::Fields;

/// Van Leer flux limiter
/// 
/// ψ(r) = (r + |r|) / (1 + |r|)
/// 
/// Gives second-order accuracy in smooth regions, first-order near discontinuities.
#[inline]
fn van_leer_limiter(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        (r + r.abs()) / (1.0 + r.abs())
    }
}

/// Compute gradient ratio for limiter
/// r = (φ_C - φ_U) / (φ_D - φ_C)
/// where U=upwind, C=center, D=downwind
#[inline]
fn gradient_ratio(phi_upwind: f64, phi_center: f64, phi_downwind: f64) -> f64 {
    let denom = phi_downwind - phi_center;
    if denom.abs() < 1e-12 {
        0.0  // Avoid division by zero, fall back to upwind
    } else {
        (phi_center - phi_upwind) / denom
    }
}

/// Advect VOF field using TVD scheme with Van Leer limiter
/// 
/// F^{n+1} = F^n - Δt * [∂(Fu)/∂x + ∂(Fv)/∂y]
/// 
/// Solid cells are treated as impermeable boundaries:
/// - No flux through solid boundaries
/// - Solid cells don't participate in advection (their VOF is unchanged)
pub fn advect_vof(
    fields: &mut Fields,
    mesh: &Mesh,
    dt: f64,
) {
    let mut vof_new = fields.vof.clone();
    
    for j in 0..mesh.ny {
        for i in 0..mesh.nx {
            // Skip solid cells - they don't participate in fluid dynamics
            // Their VOF remains unchanged (whatever was set during init)
            if fields.is_solid(i, j) {
                continue;
            }
            
            // Check for solid neighbors - these form impermeable walls
            let left_solid = i == 0 || fields.is_solid(i - 1, j);
            let right_solid = i + 1 >= mesh.nx || fields.is_solid(i + 1, j);
            let bottom_solid = j == 0 || fields.is_solid(i, j - 1);
            let top_solid = j + 1 >= mesh.ny || fields.is_solid(i, j + 1);
            
            // Get velocities
            let u_left = fields.u[[i, j]];
            let u_right = fields.u[[i + 1, j]];
            let v_bottom = fields.v[[i, j]];
            let v_top = fields.v[[i, j + 1]];
            
            // Compute fluxes - ZERO flux at solid boundaries
            // This ensures no mass transfer with solids
            
            // Left face
            let flux_left = if left_solid {
                0.0  // Solid wall: impermeable
            } else {
                let vof_face = compute_face_vof_x(fields, mesh, i, j, u_left);
                u_left * vof_face
            };
            
            // Right face
            let flux_right = if right_solid {
                0.0  // Solid wall: impermeable
            } else {
                let vof_face = compute_face_vof_x(fields, mesh, i + 1, j, u_right);
                u_right * vof_face
            };
            
            // Bottom face - critical for bathymetry!
            let flux_bottom = if bottom_solid {
                0.0  // Solid wall: impermeable (no air injection from bottom!)
            } else {
                let vof_face = compute_face_vof_y(fields, mesh, i, j, v_bottom);
                v_bottom * vof_face
            };
            
            // Top face
            let flux_top = if top_solid {
                0.0  // Solid wall: impermeable
            } else {
                let vof_face = compute_face_vof_y(fields, mesh, i, j + 1, v_top);
                v_top * vof_face
            };
            
            // Net fluxes (out - in)
            let flux_x = flux_right - flux_left;
            let flux_y = flux_top - flux_bottom;
            
            // Update VOF with divergence of flux
            vof_new[[i, j]] = fields.vof[[i, j]] 
                - dt * (flux_x / mesh.dx + flux_y / mesh.dy);
                
            // Clamp to physical range [0, 1]
            vof_new[[i, j]] = vof_new[[i, j]].clamp(0.0, 1.0);
        }
    }
    
    fields.vof = vof_new;
}

/// Compute x-direction flux with aperture scaling
/// 
/// Flux = aperture * velocity * vof_at_face
/// The aperture scales the effective area through which fluid can pass.
fn compute_flux_x_with_aperture(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
    ar_left: f64,
    ar_right: f64,
) -> f64 {
    // Get unscaled velocities for upwind determination
    let u_left = fields.u[[i, j]];
    let u_right = fields.u[[i + 1, j]];
    
    // Left face flux: ar * u * vof
    let flux_left = if ar_left > 1e-6 {
        let vof_left = compute_face_vof_x(fields, mesh, i, j, u_left);
        ar_left * u_left * vof_left
    } else {
        0.0
    };
    
    // Right face flux: ar * u * vof
    let flux_right = if ar_right > 1e-6 {
        let vof_right = compute_face_vof_x(fields, mesh, i + 1, j, u_right);
        ar_right * u_right * vof_right
    } else {
        0.0
    };
    
    // Net flux out of cell
    flux_right - flux_left
}

/// Compute y-direction flux with aperture scaling
fn compute_flux_y_with_aperture(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
    at_bottom: f64,
    at_top: f64,
) -> f64 {
    // Get unscaled velocities for upwind determination
    let v_bottom = fields.v[[i, j]];
    let v_top = fields.v[[i, j + 1]];
    
    // Bottom face flux: at * v * vof
    let flux_bottom = if at_bottom > 1e-6 {
        let vof_bottom = compute_face_vof_y(fields, mesh, i, j, v_bottom);
        at_bottom * v_bottom * vof_bottom
    } else {
        0.0
    };
    
    // Top face flux: at * v * vof
    let flux_top = if at_top > 1e-6 {
        let vof_top = compute_face_vof_y(fields, mesh, i, j + 1, v_top);
        at_top * v_top * vof_top
    } else {
        0.0
    };
    
    // Net flux out of cell
    flux_top - flux_bottom
}

/// Compute x-direction flux for cell (i, j) using TVD Van Leer
fn compute_flux_x_tvd(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    let u_left = fields.u[[i, j]];
    let u_right = fields.u[[i + 1, j]];
    
    // Left face VOF with TVD
    let vof_left = compute_face_vof_x(fields, mesh, i, j, u_left);
    
    // Right face VOF with TVD
    let vof_right = compute_face_vof_x(fields, mesh, i + 1, j, u_right);
    
    // Net flux
    u_right * vof_right - u_left * vof_left
}

/// Compute x-direction flux for cell (i, j) with solid boundary awareness
fn compute_flux_x_tvd_solid_aware(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    // Check for solid neighbors
    let left_solid = i > 0 && fields.is_solid(i - 1, j);
    let right_solid = i < mesh.nx - 1 && fields.is_solid(i + 1, j);
    
    let u_left = fields.u[[i, j]];
    let u_right = fields.u[[i + 1, j]];
    
    // Left face flux: no flux from solid
    let flux_left = if left_solid {
        0.0  // Solid boundary: no flux
    } else {
        let vof_left = compute_face_vof_x(fields, mesh, i, j, u_left);
        u_left * vof_left
    };
    
    // Right face flux: no flux into solid
    let flux_right = if right_solid {
        0.0  // Solid boundary: no flux
    } else {
        let vof_right = compute_face_vof_x(fields, mesh, i + 1, j, u_right);
        u_right * vof_right
    };
    
    flux_right - flux_left
}

/// Compute VOF at x-face using TVD scheme
fn compute_face_vof_x(
    fields: &Fields,
    mesh: &Mesh,
    face_i: usize,  // Face index (0 to nx)
    j: usize,
    u_face: f64,
) -> f64 {
    let nx = mesh.nx;
    
    // Safe cell access helper
    let get_vof = |i: usize| -> f64 {
        fields.vof[[i.min(nx - 1), j]]
    };
    
    if u_face >= 0.0 {
        // Flow to the right: upwind is left cell
        if face_i == 0 {
            // Left boundary: use boundary cell
            return get_vof(0);
        }
        
        let i_center = face_i - 1;  // Upwind cell
        let phi_c = get_vof(i_center);
        
        // Need cells at i_center-1 (upwind-upwind) and i_center+1 (downwind)
        if i_center >= 1 && i_center + 1 < nx {
            let phi_u = get_vof(i_center - 1);  // Upwind-upwind
            let phi_d = get_vof(i_center + 1);  // Downwind
            
            let r = gradient_ratio(phi_u, phi_c, phi_d);
            let psi = van_leer_limiter(r);
            
            phi_c + 0.5 * psi * (phi_d - phi_c)
        } else {
            // Near boundary: fall back to upwind
            phi_c
        }
    } else {
        // Flow to the left: upwind is right cell
        if face_i >= nx {
            // Right boundary: use boundary cell
            return get_vof(nx - 1);
        }
        
        let i_center = face_i;  // Upwind cell (right of face)
        let phi_c = get_vof(i_center);
        
        // Need cells at i_center+1 (upwind-upwind) and i_center-1 (downwind)
        if i_center >= 1 && i_center + 1 < nx {
            let phi_u = get_vof(i_center + 1);  // Upwind-upwind
            let phi_d = get_vof(i_center - 1);  // Downwind
            
            let r = gradient_ratio(phi_u, phi_c, phi_d);
            let psi = van_leer_limiter(r);
            
            phi_c + 0.5 * psi * (phi_d - phi_c)
        } else {
            phi_c
        }
    }
}

/// Compute y-direction flux for cell (i, j) using TVD Van Leer
fn compute_flux_y_tvd(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    let v_bottom = fields.v[[i, j]];
    let v_top = fields.v[[i, j + 1]];
    
    // Bottom face VOF with TVD
    let vof_bottom = compute_face_vof_y(fields, mesh, i, j, v_bottom);
    
    // Top face VOF with TVD
    let vof_top = compute_face_vof_y(fields, mesh, i, j + 1, v_top);
    
    // Net flux
    v_top * vof_top - v_bottom * vof_bottom
}

/// Compute y-direction flux for cell (i, j) with solid boundary awareness
fn compute_flux_y_tvd_solid_aware(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    // Check for solid neighbors
    let below_solid = j > 0 && fields.is_solid(i, j - 1);
    let above_solid = j < mesh.ny - 1 && fields.is_solid(i, j + 1);
    
    let v_bottom = fields.v[[i, j]];
    let v_top = fields.v[[i, j + 1]];
    
    // Bottom face flux: no flux from solid
    let flux_bottom = if below_solid {
        0.0  // Solid boundary: no flux
    } else {
        let vof_bottom = compute_face_vof_y(fields, mesh, i, j, v_bottom);
        v_bottom * vof_bottom
    };
    
    // Top face flux: no flux into solid
    let flux_top = if above_solid {
        0.0  // Solid boundary: no flux
    } else {
        let vof_top = compute_face_vof_y(fields, mesh, i, j + 1, v_top);
        v_top * vof_top
    };
    
    flux_top - flux_bottom
}

/// Compute VOF at y-face using TVD scheme
fn compute_face_vof_y(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    face_j: usize,  // Face index (0 to ny)
    v_face: f64,
) -> f64 {
    let ny = mesh.ny;
    
    // Safe cell access helper
    let get_vof = |j: usize| -> f64 {
        fields.vof[[i, j.min(ny - 1)]]
    };
    
    if v_face >= 0.0 {
        // Flow upward: upwind is bottom cell
        if face_j == 0 {
            // Bottom boundary
            return get_vof(0);
        }
        
        let j_center = face_j - 1;  // Upwind cell
        let phi_c = get_vof(j_center);
        
        // Need cells at j_center-1 (upwind-upwind) and j_center+1 (downwind)
        if j_center >= 1 && j_center + 1 < ny {
            let phi_u = get_vof(j_center - 1);  // Upwind-upwind
            let phi_d = get_vof(j_center + 1);  // Downwind
            
            let r = gradient_ratio(phi_u, phi_c, phi_d);
            let psi = van_leer_limiter(r);
            
            phi_c + 0.5 * psi * (phi_d - phi_c)
        } else {
            phi_c
        }
    } else {
        // Flow downward: upwind is top cell
        if face_j >= ny {
            // Top boundary
            return get_vof(ny - 1);
        }
        
        let j_center = face_j;  // Upwind cell (above face)
        let phi_c = get_vof(j_center);
        
        // Need cells at j_center+1 (upwind-upwind) and j_center-1 (downwind)
        if j_center >= 1 && j_center + 1 < ny {
            let phi_u = get_vof(j_center + 1);  // Upwind-upwind
            let phi_d = get_vof(j_center - 1);  // Downwind
            
            let r = gradient_ratio(phi_u, phi_c, phi_d);
            let psi = van_leer_limiter(r);
            
            phi_c + 0.5 * psi * (phi_d - phi_c)
        } else {
            phi_c
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DomainConfig;

    #[test]
    fn test_vof_conservation_no_velocity() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut fields = Fields::new(&mesh);
        
        // Set initial VOF (half water)
        for j in 0..5 {
            for i in 0..mesh.nx {
                fields.vof[[i, j]] = 1.0;
            }
        }
        
        let vol_before = fields.total_water_volume(&mesh);
        
        // Zero velocity - VOF should not change
        advect_vof(&mut fields, &mesh, 0.01);
        
        let vol_after = fields.total_water_volume(&mesh);
        
        assert!((vol_before - vol_after).abs() < 1e-10);
    }

    #[test]
    fn test_vof_clamping() {
        let mesh = Mesh::new(&DomainConfig {
            length_x: 1.0,
            length_y: 1.0,
            nx: 10,
            ny: 10,
        });
        
        let mut fields = Fields::new(&mesh);
        
        // Set some out-of-bounds values
        fields.vof[[0, 0]] = 1.5;
        fields.vof[[1, 1]] = -0.5;
        
        fields.clamp_vof();
        
        assert!(fields.vof[[0, 0]] <= 1.0);
        assert!(fields.vof[[1, 1]] >= 0.0);
    }
}

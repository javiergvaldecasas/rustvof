//! Diffusion term computation for momentum equation
//!
//! Computes ν∇²u using central differences

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// Compute diffusion term for u-velocity: ν∇²u at face (i, j)
pub fn diffuse_u(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
    nu: f64,
) -> f64 {
    let u_ij = fields.u[[i, j]];
    
    // d²u/dx² using central differences
    let d2u_dx2 = if i > 0 && i < mesh.nx {
        (fields.u[[i + 1, j]] - 2.0 * u_ij + fields.u[[i - 1, j]]) / (mesh.dx * mesh.dx)
    } else if i == 0 {
        // Left boundary - one-sided
        (fields.u[[i + 1, j]] - u_ij) / (mesh.dx * mesh.dx)
    } else {
        // Right boundary - one-sided
        (fields.u[[i - 1, j]] - u_ij) / (mesh.dx * mesh.dx)
    };
    
    // d²u/dy² using central differences
    let d2u_dy2 = if j > 0 && j < mesh.ny - 1 {
        (fields.u[[i, j + 1]] - 2.0 * u_ij + fields.u[[i, j - 1]]) / (mesh.dy * mesh.dy)
    } else if j == 0 {
        // Bottom boundary
        (fields.u[[i, j + 1]] - u_ij) / (mesh.dy * mesh.dy)
    } else {
        // Top boundary
        (fields.u[[i, j - 1]] - u_ij) / (mesh.dy * mesh.dy)
    };
    
    nu * (d2u_dx2 + d2u_dy2)
}

/// Compute diffusion term for v-velocity: ν∇²v at face (i, j)
pub fn diffuse_v(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
    nu: f64,
) -> f64 {
    let v_ij = fields.v[[i, j]];
    
    // d²v/dx² using central differences
    let d2v_dx2 = if i > 0 && i < mesh.nx - 1 {
        (fields.v[[i + 1, j]] - 2.0 * v_ij + fields.v[[i - 1, j]]) / (mesh.dx * mesh.dx)
    } else if i == 0 {
        (fields.v[[i + 1, j]] - v_ij) / (mesh.dx * mesh.dx)
    } else {
        (fields.v[[i - 1, j]] - v_ij) / (mesh.dx * mesh.dx)
    };
    
    // d²v/dy² using central differences
    let d2v_dy2 = if j > 0 && j < mesh.ny {
        (fields.v[[i, j + 1]] - 2.0 * v_ij + fields.v[[i, j - 1]]) / (mesh.dy * mesh.dy)
    } else if j == 0 {
        (fields.v[[i, j + 1]] - v_ij) / (mesh.dy * mesh.dy)
    } else {
        (fields.v[[i, j - 1]] - v_ij) / (mesh.dy * mesh.dy)
    };
    
    nu * (d2v_dx2 + d2v_dy2)
}

/// Compute diffusion terms for all velocity components
/// Returns (diff_u, diff_v) arrays
pub fn compute_diffusion(
    fields: &Fields,
    mesh: &Mesh,
    nu: f64,
) -> (Array2<f64>, Array2<f64>) {
    let (nu_x, nu_y) = mesh.num_u();
    let (nv_x, nv_y) = mesh.num_v();
    
    let mut diff_u = Array2::zeros((nu_x, nu_y));
    let mut diff_v = Array2::zeros((nv_x, nv_y));
    
    // All u points
    for j in 0..nu_y {
        for i in 0..nu_x {
            diff_u[[i, j]] = diffuse_u(fields, mesh, i, j, nu);
        }
    }
    
    // All v points
    for j in 0..nv_y {
        for i in 0..nv_x {
            diff_v[[i, j]] = diffuse_v(fields, mesh, i, j, nu);
        }
    }
    
    (diff_u, diff_v)
}

/// Compute diffusion with VOF-based treatment
/// 
/// In air (VOF < threshold): no diffusion (air is essentially inviscid)
/// In water (VOF >= threshold): use water viscosity
/// 
/// This is a simplification. The full form would be (1/ρ)∇·(μ∇u).
pub fn compute_diffusion_variable(
    fields: &Fields,
    mesh: &Mesh,
) -> (Array2<f64>, Array2<f64>) {
    let (nu_x, nu_y) = mesh.num_u();
    let (nv_x, nv_y) = mesh.num_v();
    
    let mut diff_u = Array2::zeros((nu_x, nu_y));
    let mut diff_v = Array2::zeros((nv_x, nv_y));
    
    const VOF_THRESHOLD: f64 = 0.01;
    const NU_WATER: f64 = 1.0e-6;  // Water kinematic viscosity
    
    // u points - only apply diffusion where there's water
    // Skip solid cells (e.g., behind moving paddle)
    for j in 0..nu_y {
        for i in 0..nu_x {
            // u-face (i,j) is between cells (i-1,j) and (i,j)
            // Skip if either cell is solid
            let left_solid = i > 0 && i <= mesh.nx && fields.is_solid(i - 1, j);
            let right_solid = i < mesh.nx && fields.is_solid(i, j);
            
            if left_solid || right_solid {
                diff_u[[i, j]] = 0.0;
                continue;
            }
            
            // Interpolate VOF to u-face
            let vof_face = if i > 0 && i < mesh.nx {
                0.5 * (fields.vof[[i - 1, j]] + fields.vof[[i, j]])
            } else if i == 0 {
                fields.vof[[0, j]]
            } else {
                fields.vof[[mesh.nx - 1, j]]
            };
            
            if vof_face >= VOF_THRESHOLD {
                diff_u[[i, j]] = diffuse_u(fields, mesh, i, j, NU_WATER);
            } else {
                diff_u[[i, j]] = 0.0;
            }
        }
    }

    // v points - only apply diffusion where there's water (separate loop!)
    // Skip solid cells
    for j in 0..nv_y {
        for i in 0..nv_x {
            // v-face (i,j) is between cells (i,j-1) and (i,j)
            // Skip if either cell is solid
            let below_solid = j > 0 && j <= mesh.ny && fields.is_solid(i, j - 1);
            let above_solid = j < mesh.ny && fields.is_solid(i, j);
            
            if below_solid || above_solid {
                diff_v[[i, j]] = 0.0;
                continue;
            }
            
            // Interpolate VOF to v-face
            let vof_face = if j > 0 && j < mesh.ny {
                0.5 * (fields.vof[[i, j - 1]] + fields.vof[[i, j]])
            } else if j == 0 {
                fields.vof[[i, 0]]
            } else {
                fields.vof[[i, mesh.ny - 1]]
            };
            
            if vof_face >= VOF_THRESHOLD {
                diff_v[[i, j]] = diffuse_v(fields, mesh, i, j, NU_WATER);
            } else {
                diff_v[[i, j]] = 0.0;
            }
        }
    }
    
    (diff_u, diff_v)
}

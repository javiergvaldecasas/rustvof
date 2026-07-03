//! Advection term computation for momentum equation
//!
//! Computes (u·∇)u using upwind differencing for stability

use ndarray::Array2;
use crate::mesh::Mesh;
use crate::fields::Fields;

/// Compute advection term for u-velocity: (u·∇)u at face (i, j)
pub fn advect_u(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    // u at this face
    let u_ij = fields.u[[i, j]];
    
    // Interpolated v at this u-location
    let v_at_u = if i == 0 || i == mesh.nx {
        // Boundary - use nearest v
        if j < mesh.ny {
            0.5 * (fields.v[[i.min(mesh.nx - 1), j]] + fields.v[[i.min(mesh.nx - 1), j + 1]])
        } else {
            0.0
        }
    } else {
        // Interior - average 4 surrounding v values
        0.25 * (
            fields.v[[i - 1, j]] + fields.v[[i - 1, j + 1]] +
            fields.v[[i, j]] + fields.v[[i, j + 1]]
        )
    };

    // d(uu)/dx using upwind
    let du_dx = if u_ij >= 0.0 {
        // Upwind from left
        if i > 0 {
            (u_ij * u_ij - fields.u[[i - 1, j]] * fields.u[[i - 1, j]]) / mesh.dx
        } else {
            0.0
        }
    } else {
        // Upwind from right
        if i < mesh.nx {
            (fields.u[[i + 1, j]] * fields.u[[i + 1, j]] - u_ij * u_ij) / mesh.dx
        } else {
            0.0
        }
    };

    // d(uv)/dy using upwind
    let duv_dy = if v_at_u >= 0.0 {
        // Upwind from below
        if j > 0 {
            let u_below = fields.u[[i, j - 1]];
            let v_below = if i == 0 || i == mesh.nx {
                fields.v[[i.min(mesh.nx - 1), j]]
            } else {
                0.5 * (fields.v[[i - 1, j]] + fields.v[[i, j]])
            };
            (u_ij * v_at_u - u_below * v_below) / mesh.dy
        } else {
            0.0
        }
    } else {
        // Upwind from above
        if j < mesh.ny - 1 {
            let u_above = fields.u[[i, j + 1]];
            let v_above = if i == 0 || i == mesh.nx {
                fields.v[[i.min(mesh.nx - 1), j + 1]]
            } else {
                0.5 * (fields.v[[i - 1, j + 1]] + fields.v[[i, j + 1]])
            };
            (u_above * v_above - u_ij * v_at_u) / mesh.dy
        } else {
            0.0
        }
    };

    du_dx + duv_dy
}

/// Compute advection term for v-velocity: (u·∇)v at face (i, j)
pub fn advect_v(
    fields: &Fields,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> f64 {
    // v at this face
    let v_ij = fields.v[[i, j]];
    
    // Interpolated u at this v-location
    let u_at_v = if j == 0 || j == mesh.ny {
        // Boundary
        0.5 * (fields.u[[i, j.min(mesh.ny - 1)]] + fields.u[[i + 1, j.min(mesh.ny - 1)]])
    } else {
        // Interior - average 4 surrounding u values
        0.25 * (
            fields.u[[i, j - 1]] + fields.u[[i + 1, j - 1]] +
            fields.u[[i, j]] + fields.u[[i + 1, j]]
        )
    };

    // d(uv)/dx using upwind
    let duv_dx = if u_at_v >= 0.0 {
        if i > 0 {
            let v_left = fields.v[[i - 1, j]];
            let u_left = if j == 0 || j == mesh.ny {
                fields.u[[i, j.min(mesh.ny - 1)]]
            } else {
                0.5 * (fields.u[[i, j - 1]] + fields.u[[i, j]])
            };
            (u_at_v * v_ij - u_left * v_left) / mesh.dx
        } else {
            0.0
        }
    } else {
        if i < mesh.nx - 1 {
            let v_right = fields.v[[i + 1, j]];
            let u_right = if j == 0 || j == mesh.ny {
                fields.u[[i + 1, j.min(mesh.ny - 1)]]
            } else {
                0.5 * (fields.u[[i + 1, j - 1]] + fields.u[[i + 1, j]])
            };
            (u_right * v_right - u_at_v * v_ij) / mesh.dx
        } else {
            0.0
        }
    };

    // d(vv)/dy using upwind
    let dv_dy = if v_ij >= 0.0 {
        if j > 0 {
            (v_ij * v_ij - fields.v[[i, j - 1]] * fields.v[[i, j - 1]]) / mesh.dy
        } else {
            0.0
        }
    } else {
        if j < mesh.ny {
            (fields.v[[i, j + 1]] * fields.v[[i, j + 1]] - v_ij * v_ij) / mesh.dy
        } else {
            0.0
        }
    };

    duv_dx + dv_dy
}

/// Compute advection terms for all velocity components
/// Returns (adv_u, adv_v) arrays
/// 
/// Skips solid cells (e.g., moving paddle region)
pub fn compute_advection(
    fields: &Fields,
    mesh: &Mesh,
) -> (Array2<f64>, Array2<f64>) {
    let (nu_x, nu_y) = mesh.num_u();
    let (nv_x, nv_y) = mesh.num_v();
    
    let mut adv_u = Array2::zeros((nu_x, nu_y));
    let mut adv_v = Array2::zeros((nv_x, nv_y));
    
    // Interior u points (not on left/right boundary)
    // Skip if neighboring cells are solid (e.g., behind paddle)
    for j in 0..nu_y {
        for i in 1..nu_x - 1 {
            // u-face (i,j) is between cells (i-1,j) and (i,j)
            // Skip if either cell is solid
            let left_solid = i > 0 && fields.is_solid(i - 1, j);
            let right_solid = i < mesh.nx && fields.is_solid(i, j);

            if !left_solid && !right_solid {
                adv_u[[i, j]] = advect_u(fields, mesh, i, j);
            }
            // else: adv_u remains 0.0 (no advection in/out of solid)

            // v-face (i,j) is between cells (i,j-1) and (i,j)
            // Skip if either cell is solid
            let below_solid = j > 0 && fields.is_solid(i, j - 1);
            let above_solid = j < mesh.ny && fields.is_solid(i, j);

            if !below_solid && !above_solid {
                adv_v[[i, j]] = advect_v(fields, mesh, i, j);
            }
        }
    }


    (adv_u, adv_v)
}

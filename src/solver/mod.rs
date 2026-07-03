//! Numerical solvers for Navier-Stokes and VOF equations

pub mod advection;
pub mod diffusion;
pub mod pressure;
pub mod projection;
pub mod turbulence;
pub mod vof;

pub use advection::compute_advection;
pub use diffusion::{compute_diffusion, compute_diffusion_variable};
pub use pressure::PressureSolver;
pub use projection::{project_velocity, project_velocity_variable_density, enforce_solid_bc};
pub use turbulence::KEpsilonModel;
pub use vof::advect_vof;

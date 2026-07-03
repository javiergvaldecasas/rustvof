//! VOF2D - 2D Volume of Fluid solver for wave simulation
//!
//! A numerical solver for the incompressible Navier-Stokes equations
//! with free surface tracking using the Volume of Fluid (VOF) method.
//!
//! ## GPU Acceleration
//! 
//! This solver supports GPU acceleration via wgpu (Metal on Apple Silicon,
//! Vulkan/DX12 on other platforms). Enable with the `gpu` feature (default).
//! 
//! ```ignore
//! use rustvof::gpu::GpuContext;
//! 
//! let gpu = GpuContext::new(&mesh)?;
//! gpu.upload_fields(&fields);
//! gpu.solve_pressure(&mesh, dt, max_iter, tolerance);
//! gpu.download_fields(&mut fields);
//! ```

pub mod config;
pub mod mesh;
pub mod cell_type;
pub mod fields;
pub mod properties;
pub mod solver;
pub mod boundary;
pub mod output;
pub mod simulation;

pub use config::Config;
pub use mesh::Mesh;
pub use cell_type::{CellType, PorousProperties};
pub use fields::Fields;
pub use simulation::Simulation;



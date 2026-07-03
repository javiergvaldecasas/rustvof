//! Boundary condition implementations

pub mod walls;
pub mod waves;

pub use walls::apply_wall_bc;
pub use waves::{WaveGenerator, SolitaryWaveGenerator};

//! Output writers for simulation results

pub mod vtk;
pub mod probes;

#[cfg(feature = "netcdf-output")]
pub mod netcdf;

pub use vtk::VtkWriter;
pub use probes::ProbeWriter;

#[cfg(feature = "netcdf-output")]
pub use self::netcdf::NetcdfWriter;

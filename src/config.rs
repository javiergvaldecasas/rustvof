//! Configuration parsing from TOML files

use serde::Deserialize;
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub domain: DomainConfig,
    pub time: TimeConfig,
    pub fluid: FluidConfig,
    pub initial_condition: InitialConditionConfig,
    #[serde(default)]
    pub wave: WaveConfig,
    #[serde(default)]
    pub bathymetry: BathymetryConfig,
    pub output: OutputConfig,
    /// Probe locations for time series output
    #[serde(default)]
    pub probes: ProbesConfig,
    /// Solver settings
    #[serde(default)]
    pub solver: SolverConfig,
    /// Turbulence model settings
    #[serde(default)]
    pub turbulence: TurbulenceConfig,
}

/// Pressure solver type
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PressureSolverType {
    /// Iterative SOR solver (works with any geometry)
    #[default]
    Iterative,
}

/// Solver configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SolverConfig {
    /// Pressure solver type
    #[serde(default)]
    pub pressure_solver: PressureSolverType,
    /// Maximum iterations for iterative solvers
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    /// Convergence tolerance
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// SOR relaxation factor (1.0 = Jacobi, 1.5-1.9 typical for SOR)
    #[serde(default = "default_omega")]
    pub omega: f64,
}

fn default_max_iter() -> usize { 10000 }
fn default_tolerance() -> f64 { 1e-6 }
fn default_omega() -> f64 { 1.5 }

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            pressure_solver: PressureSolverType::default(),
            max_iter: default_max_iter(),
            tolerance: default_tolerance(),
            omega: default_omega(),
        }
    }
}

/// Turbulence model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct TurbulenceConfig {
    /// Enable turbulence modeling
    #[serde(default)]
    pub enabled: bool,
    /// Turbulence model type (only "k-epsilon" supported for now)
    #[serde(default = "default_turbulence_model")]
    pub model: String,
    /// Initial turbulence intensity (typically 0.01-0.1 for 1%-10%)
    #[serde(default = "default_turbulence_intensity")]
    pub intensity: f64,
    /// Turbulent length scale (typically 0.07 * hydraulic diameter)
    #[serde(default = "default_length_scale")]
    pub length_scale: f64,
}

fn default_turbulence_model() -> String { "k-epsilon".to_string() }
fn default_turbulence_intensity() -> f64 { 0.05 }  // 5%
fn default_length_scale() -> f64 { 0.1 }  // 10cm

impl Default for TurbulenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_turbulence_model(),
            intensity: default_turbulence_intensity(),
            length_scale: default_length_scale(),
        }
    }
}

/// Domain configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DomainConfig {
    /// Domain length in x direction (meters)
    pub length_x: f64,
    /// Domain length in y direction (meters)
    pub length_y: f64,
    /// Number of cells in x direction
    pub nx: usize,
    /// Number of cells in y direction
    pub ny: usize,
}

impl DomainConfig {
    /// Cell size in x direction
    pub fn dx(&self) -> f64 {
        self.length_x / self.nx as f64
    }

    /// Cell size in y direction
    pub fn dy(&self) -> f64 {
        self.length_y / self.ny as f64
    }
}

/// Time configuration
#[derive(Debug, Clone, Deserialize)]
pub struct TimeConfig {
    /// End time (seconds)
    pub t_end: f64,
    /// Output interval (seconds)
    pub dt_output: f64,
    /// CFL number for adaptive time stepping
    #[serde(default = "default_cfl")]
    pub cfl: f64,
    /// Maximum time step (optional)
    pub dt_max: Option<f64>,
}

fn default_cfl() -> f64 {
    0.5
}

/// Fluid properties
#[derive(Debug, Clone, Deserialize)]
pub struct FluidConfig {
    /// Water density (kg/m³)
    #[serde(default = "default_rho_water")]
    pub rho_water: f64,
    /// Air density (kg/m³)
    #[serde(default = "default_rho_air")]
    pub rho_air: f64,
    /// Water kinematic viscosity (m²/s)
    #[serde(default = "default_nu_water")]
    pub nu_water: f64,
    /// Air kinematic viscosity (m²/s)
    #[serde(default = "default_nu_air")]
    pub nu_air: f64,
    /// Gravitational acceleration (m/s²)
    #[serde(default = "default_gravity")]
    pub gravity: f64,
}

fn default_rho_water() -> f64 { 1000.0 }
fn default_rho_air() -> f64 { 1.225 }
fn default_nu_water() -> f64 { 1.0e-6 }
fn default_nu_air() -> f64 { 1.5e-5 }
fn default_gravity() -> f64 { 9.81 }

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            rho_water: default_rho_water(),
            rho_air: default_rho_air(),
            nu_water: default_nu_water(),
            nu_air: default_nu_air(),
            gravity: default_gravity(),
        }
    }
}

/// Initial condition configuration
#[derive(Debug, Clone, Deserialize)]
pub struct InitialConditionConfig {
    /// Initial water level from bottom (meters)
    pub water_level: f64,
}

/// Wave type for generation
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WaveType {
    /// Regular periodic wave (Stokes/linear theory)
    #[default]
    Periodic,
    /// Solitary wave (Boussinesq theory)
    Solitary,
}

/// Wave generation configuration
#[derive(Debug, Clone, Deserialize)]
pub struct WaveConfig {
    /// Enable wave generation
    #[serde(default)]
    pub enabled: bool,
    /// Wave type: "periodic" or "solitary"
    #[serde(default, rename = "type")]
    pub wave_type: WaveType,
    /// Wave height H for periodic waves (meters)
    #[serde(default)]
    pub height: f64,
    /// Wave amplitude a for solitary waves (meters)
    #[serde(default)]
    pub amplitude: f64,
    /// Wave period T for periodic waves (seconds)
    #[serde(default)]
    pub period: f64,
    /// Water depth for wave theory (meters)
    #[serde(default)]
    pub depth: f64,
    /// Initial paddle position x0 (meters from left wall)
    /// The paddle oscillates around this position.
    /// Set to 0 for fixed boundary (legacy mode).
    #[serde(default)]
    pub paddle_x0: f64,
    /// X position for wave generation (solitary wave)
    #[serde(default)]
    pub generation_x: f64,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            wave_type: WaveType::Periodic,
            height: 0.0,
            amplitude: 0.0,
            period: 1.0,
            depth: 1.0,
            paddle_x0: 0.0,
            generation_x: 0.0,
        }
    }
}

impl WaveConfig {
    /// Check if this is a solitary wave
    pub fn is_solitary(&self) -> bool {
        self.wave_type == WaveType::Solitary
    }

    /// Get effective wave amplitude
    /// For periodic waves: H/2
    /// For solitary waves: amplitude
    pub fn wave_amplitude(&self) -> f64 {
        if self.is_solitary() {
            self.amplitude
        } else {
            self.height / 2.0
        }
    }

    /// Solitary wave celerity: c = sqrt(g(h + a))
    pub fn solitary_celerity(&self, g: f64) -> f64 {
        (g * (self.depth + self.amplitude)).sqrt()
    }

    /// Solitary wave characteristic width parameter k
    /// k = sqrt(3a / 4h³)
    pub fn solitary_k(&self) -> f64 {
        (3.0 * self.amplitude / (4.0 * self.depth.powi(3))).sqrt()
    }
}

/// Bathymetry configuration for bottom topography
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BathymetryConfig {
    /// Bathymetry type: "flat", "slope", or "custom"
    #[serde(default, rename = "type")]
    pub bathy_type: BathymetryType,
    /// X coordinate where slope starts
    #[serde(default)]
    pub slope_start_x: f64,
    /// Y coordinate (bottom elevation) at slope start
    #[serde(default)]
    pub slope_start_y: f64,
    /// X coordinate where slope ends
    #[serde(default)]
    pub slope_end_x: f64,
    /// Y coordinate (bottom elevation) at slope end
    #[serde(default)]
    pub slope_end_y: f64,
}

/// Bathymetry type
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BathymetryType {
    /// Flat bottom (default)
    #[default]
    Flat,
    /// Linear slope
    Slope,
    /// Custom bathymetry from file (not yet implemented)
    Custom,
}

impl BathymetryConfig {
    /// Get bottom elevation at x position
    /// Returns the y coordinate of the bottom (0 = flat bottom)
    /// 
    /// For slope bathymetry:
    /// - x < slope_start_x: flat bottom at y = 0
    /// - x >= slope_start_x and x <= slope_end_x: linear slope from start_y to end_y
    /// - x > slope_end_x: constant at end_y
    pub fn bottom_at(&self, x: f64) -> f64 {
        match self.bathy_type {
            BathymetryType::Flat => 0.0,
            BathymetryType::Slope => {
                if x < self.slope_start_x {
                    // Flat bottom before slope
                    0.0
                } else if x >= self.slope_end_x {
                    // Constant elevation after slope ends
                    self.slope_end_y
                } else {
                    // Linear interpolation on the slope
                    let t = (x - self.slope_start_x) / (self.slope_end_x - self.slope_start_x);
                    self.slope_start_y + t * (self.slope_end_y - self.slope_start_y)
                }
            }
            BathymetryType::Custom => 0.0, // TODO: implement
        }
    }

    /// Check if a cell (i, j) is inside the bathymetry (solid)
    pub fn is_solid(&self, x: f64, y: f64) -> bool {
        y < self.bottom_at(x)
    }
}

impl WaveConfig {
    /// Calculate wave number k from dispersion relation
    /// ω² = gk·tanh(kd)
    pub fn wave_number(&self, g: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI / self.period;
        let omega2 = omega * omega;
        
        // Newton-Raphson iteration for k
        let mut k = omega2 / g; // Deep water approximation as initial guess
        
        for _ in 0..50 {
            let f = omega2 - g * k * (k * self.depth).tanh();
            let df = -g * ((k * self.depth).tanh() + k * self.depth / (k * self.depth).cosh().powi(2));
            let dk = f / df;
            k -= dk;
            if dk.abs() < 1e-10 {
                break;
            }
        }
        k
    }

    /// Angular frequency
    pub fn omega(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.period
    }

    /// Wavelength
    pub fn wavelength(&self, g: f64) -> f64 {
        2.0 * std::f64::consts::PI / self.wave_number(g)
    }
}

/// Output format enum
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// VTK format (for ParaView)
    #[default]
    Vtk,
    /// NetCDF format (CF-compliant)
    Netcdf,
    /// Both VTK and NetCDF
    Both,
}

/// Output configuration
#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    /// Output format: "vtk", "netcdf", or "both"
    #[serde(default)]
    pub format: OutputFormat,
    /// Output file prefix (for VTK: prefix_000001.vtk, for NetCDF: prefix.nc)
    pub prefix: String,
    /// Number of frames per NetCDF file (0 = single file at end)
    /// When set, creates prefix_000.nc, prefix_001.nc, etc.
    #[serde(default)]
    pub frames_per_file: usize,
}

/// Probes configuration for time series output
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProbesConfig {
    /// Enable probe output
    #[serde(default)]
    pub enabled: bool,
    /// Output file path for probe data (CSV)
    #[serde(default = "default_probe_file")]
    pub file: String,
    /// Output interval (seconds). If not set, uses dt_output
    pub dt: Option<f64>,
    /// Wave gauge locations (x coordinates in meters)
    #[serde(default)]
    pub wave_gauges: Vec<f64>,
    // Future: velocity probes, pressure probes
    // pub velocity_probes: Vec<ProbeLocation>,
    // pub pressure_probes: Vec<ProbeLocation>,
}

fn default_probe_file() -> String {
    "probes.csv".to_string()
}

/// A probe location in 2D space
#[derive(Debug, Clone, Deserialize)]
pub struct ProbeLocation {
    pub x: f64,
    pub y: f64,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Create a simple sloshing tank configuration for testing
    pub fn sloshing_tank() -> Self {
        Config {
            domain: DomainConfig {
                length_x: 1.0,
                length_y: 0.6,
                nx: 100,
                ny: 60,
            },
            time: TimeConfig {
                t_end: 5.0,
                dt_output: 0.05,
                cfl: 0.5,
                dt_max: Some(0.001),
            },
            fluid: FluidConfig::default(),
            initial_condition: InitialConditionConfig {
                water_level: 0.3,
            },
            wave: WaveConfig::default(),
            bathymetry: BathymetryConfig::default(),
            output: OutputConfig {
                format: OutputFormat::Vtk,
                prefix: "output/sloshing".to_string(),
                frames_per_file: 0,
            },
            probes: ProbesConfig::default(),
            solver: SolverConfig::default(),
            turbulence: TurbulenceConfig::default(),
        }
    }
}


//! Wave generation boundary conditions
//!
//! Implements piston-type wavemaker with moving paddle and solitary wave generation.
//! 
//! Two modes:
//! - Periodic waves: piston wavemaker with paddle
//! - Solitary wave: Boussinesq theory with inlet boundary condition

use std::f64::consts::PI;
use crate::mesh::Mesh;
use crate::fields::Fields;
use crate::config::{WaveConfig, WaveType};

/// Piston-type wave generator with moving paddle
/// 
/// The paddle oscillates around position x0, creating waves that
/// propagate to the right. Cells to the left of the paddle are
/// treated as solid (blocked).
/// 
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// │  WALL │ SOLID      │ PADDLE │      FLUID (waves →)    │    │
/// │       │ (blocked)  │        │                          │    │
/// ├───────┼────────────┼────────┼──────────────────────────┼────┤
/// │  x=0  │            │ x_paddle(t)                       │x=L │
/// └─────────────────────────────────────────────────────────────┘
///         ←── stroke ──→
/// ```
pub struct WaveGenerator {
    /// Wave height H (m)
    pub height: f64,
    /// Wave period T (s)
    pub period: f64,
    /// Water depth d (m)
    pub depth: f64,
    /// Wave number k (rad/m)
    pub k: f64,
    /// Angular frequency ω (rad/s)
    pub omega: f64,
    /// Gravitational acceleration (m/s²)
    pub gravity: f64,
    /// Paddle stroke S (m) - peak to peak displacement
    pub stroke: f64,
    /// Initial paddle position x0 (m)
    pub x0: f64,
    /// Ramp-up duration in periods (default: 2.0)
    pub ramp_periods: f64,
}

impl WaveGenerator {
    /// Create a new wave generator from configuration
    pub fn new(config: &WaveConfig, gravity: f64) -> Self {
        let omega = 2.0 * PI / config.period;
        let k = config.wave_number(gravity);
        
        // Piston wavemaker transfer function:
        // H/S = 2(cosh(2kd) - 1) / (sinh(2kd) + 2kd)
        let kd = k * config.depth;
        let two_kd = 2.0 * kd;
        let transfer = (two_kd.sinh() + two_kd) / (2.0 * (two_kd.cosh() - 1.0));
        let stroke = config.height * transfer;
        
        Self {
            height: config.height,
            period: config.period,
            depth: config.depth,
            k,
            omega,
            gravity,
            stroke,
            x0: config.paddle_x0,
            ramp_periods: 2.0,  // Default: 2 periods ramp-up
        }
    }

    /// Calculate ramp-up gain factor at time t
    /// 
    /// Returns a value from 0 to 1 that smoothly increases
    /// over the first `ramp_periods` wave periods.
    /// 
    /// Uses cosine ramp: gain = 0.5 * (1 - cos(π * t / T_ramp))
    pub fn ramp_gain(&self, t: f64) -> f64 {
        if self.ramp_periods <= 0.0 {
            return 1.0;
        }
        
        let t_ramp = self.ramp_periods * self.period;
        
        if t >= t_ramp {
            1.0
        } else if t <= 0.0 {
            0.0
        } else {
            0.5 * (1.0 - (PI * t / t_ramp).cos())
        }
    }

    /// Calculate wavelength
    pub fn wavelength(&self) -> f64 {
        2.0 * PI / self.k
    }

    /// Calculate phase velocity
    pub fn phase_velocity(&self) -> f64 {
        self.omega / self.k
    }

    /// Calculate paddle position at time t (with ramp-up)
    /// x_paddle(t) = x0 + gain(t) * (S/2) * sin(ωt)
    pub fn paddle_position(&self, t: f64) -> f64 {
        let gain = self.ramp_gain(t);
        self.x0 + gain * 0.5 * self.stroke * (self.omega * t).sin()
    }

    /// Calculate paddle velocity at time t (with ramp-up)
    /// 
    /// Full derivative: d/dt[gain(t) * (S/2) * sin(ωt)]
    ///   = gain'(t) * (S/2) * sin(ωt) + gain(t) * (S/2) * ω * cos(ωt)
    /// 
    /// For smooth motion, we include both terms.
    pub fn paddle_velocity(&self, t: f64) -> f64 {
        let gain = self.ramp_gain(t);
        
        // Derivative of gain: gain'(t) = (π / 2T_ramp) * sin(π * t / T_ramp)
        let t_ramp = self.ramp_periods * self.period;
        let gain_derivative = if t > 0.0 && t < t_ramp && self.ramp_periods > 0.0 {
            (PI / (2.0 * t_ramp)) * (PI * t / t_ramp).sin()
        } else {
            0.0
        };
        
        let sin_wt = (self.omega * t).sin();
        let cos_wt = (self.omega * t).cos();
        
        // u = gain' * (S/2) * sin(ωt) + gain * (S/2) * ω * cos(ωt)
        gain_derivative * 0.5 * self.stroke * sin_wt 
            + gain * 0.5 * self.stroke * self.omega * cos_wt
    }

    /// Calculate paddle acceleration at time t (with ramp-up)
    pub fn paddle_acceleration(&self, t: f64) -> f64 {
        let gain = self.ramp_gain(t);
        
        // Simplified: use ramped acceleration
        -gain * 0.5 * self.stroke * self.omega * self.omega * (self.omega * t).sin()
    }

    /// Check if a point x is in the solid region (behind paddle)
    pub fn is_solid(&self, x: f64, t: f64) -> bool {
        x < self.paddle_position(t)
    }

    /// Get the cell index where the paddle is located
    pub fn paddle_cell_index(&self, mesh: &Mesh, t: f64) -> usize {
        let x_paddle = self.paddle_position(t);
        let i = (x_paddle / mesh.dx).floor() as usize;
        i.min(mesh.nx - 1)
    }

    /// Apply moving paddle boundary conditions
    /// 
    /// This method:
    /// 1. Zeros velocity in solid cells (behind paddle)
    /// 2. Applies paddle velocity at paddle face
    /// 3. Sets VOF correctly in solid region (water below still level)
    /// 4. Imposes v=0 at paddle face (impermeability)
    pub fn apply_moving_paddle(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
        still_water_level: f64,
    ) {
        let x_paddle = self.paddle_position(t);
        let u_paddle = self.paddle_velocity(t);
        
        // Find the u-face index just to the right of the paddle
        // u-faces are at x = i * dx, so i_face is the first face with x >= x_paddle
        let i_face = ((x_paddle / mesh.dx).ceil() as usize).min(mesh.nx);
        
        // Cell index of the paddle (cell containing the paddle)
        let i_paddle_cell = (x_paddle / mesh.dx).floor() as usize;
        let i_paddle_cell = i_paddle_cell.min(mesh.nx - 1);
        
        // Get water level from VOF just ahead of paddle
        let i_ahead = (i_paddle_cell + 1).min(mesh.nx - 1);
        let water_level = fields.eta_at(i_ahead, mesh);
        
        // 1. Zero velocity in solid region (all faces with x < x_paddle)
        for j in 0..mesh.ny {
            for i in 0..i_face {
                fields.u[[i, j]] = 0.0;
            }
        }
        // Zero v in solid region
        for j in 0..=mesh.ny {
            for i in 0..i_paddle_cell {
                fields.v[[i, j]] = 0.0;
            }
        }
        
        // 2. Apply paddle velocity at the paddle face (u)
        for j in 0..mesh.ny {
            let y_center = mesh.y_center(j);
            
            if y_center <= water_level {
                // Below water: apply paddle velocity
                fields.u[[i_face, j]] = u_paddle;
            } else {
                // Above water: no horizontal velocity at paddle
                fields.u[[i_face, j]] = 0.0;
            }
        }
        
        // 3. Set VOF in solid region based on still water level
        // This ensures when paddle retracts, cells have correct VOF
        for i in 0..=i_paddle_cell {
            for j in 0..mesh.ny {
                let y_center = mesh.y_center(j);
                let y_bottom = mesh.y_face(j);
                let y_top = mesh.y_face(j + 1);
                
                if y_top <= still_water_level {
                    // Fully below water
                    fields.vof[[i, j]] = 1.0;
                } else if y_bottom >= still_water_level {
                    // Fully above water
                    fields.vof[[i, j]] = 0.0;
                } else {
                    // Partially filled
                    fields.vof[[i, j]] = (still_water_level - y_bottom) / mesh.dy;
                }
            }
        }
        
        // 4. Impose v=0 at paddle cell (impermeability in y)
        // The paddle is a vertical wall, so v should be zero at its location
        for j in 0..=mesh.ny {
            if i_paddle_cell < mesh.nx {
                fields.v[[i_paddle_cell, j]] = 0.0;
            }
        }
    }

    /// Legacy method: Apply boundary at fixed left wall (x=0)
    /// Used when paddle_x0 = 0
    pub fn apply_boundary(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
        still_water_level: f64,
    ) {
        if self.x0 > 0.0 {
            // Use moving paddle
            self.apply_moving_paddle(fields, mesh, t, still_water_level);
        } else {
            // Legacy: fixed boundary at x=0
            self.apply_fixed_boundary(fields, mesh, t, still_water_level);
        }
    }

    /// Apply piston wavemaker at fixed left boundary (legacy)
    fn apply_fixed_boundary(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
        _still_water_level: f64,
    ) {
        let u_paddle = self.paddle_velocity(t);
        
        // Get water level from VOF solution at boundary
        // Use first interior column to get the actual free surface
        let water_level = fields.eta_at(1.min(mesh.nx - 1), mesh);
        
        // Apply horizontal velocity at left boundary (i = 0)
        for j in 0..mesh.ny {
            let y_center = mesh.y_center(j);
            
            if y_center <= water_level {
                fields.u[[0, j]] = u_paddle;
            } else {
                fields.u[[0, j]] = 0.0;
            }
        }
        
        // Vertical velocity at paddle is zero
        for j in 0..=mesh.ny {
            fields.v[[0, j]] = 0.0;
        }
        
        // Update VOF at left boundary
        for j in 0..mesh.ny {
            let y_bottom = mesh.y_face(j);
            let y_top = mesh.y_face(j + 1);
            
            if y_top <= water_level {
                fields.vof[[0, j]] = 1.0;
            } else if y_bottom >= water_level {
                fields.vof[[0, j]] = 0.0;
            } else {
                fields.vof[[0, j]] = (water_level - y_bottom) / mesh.dy;
            }
        }
    }

    /// Get solid mask for pressure solver
    /// Returns true for cells that should be excluded from pressure solve
    pub fn get_solid_mask(&self, mesh: &Mesh, t: f64) -> Vec<Vec<bool>> {
        let x_paddle = self.paddle_position(t);
        let mut mask = vec![vec![false; mesh.ny]; mesh.nx];
        
        for i in 0..mesh.nx {
            let x_center = mesh.x_center(i);
            if x_center < x_paddle {
                for j in 0..mesh.ny {
                    mask[i][j] = true;
                }
            }
        }
        
        mask
    }
}

/// Solitary wave generator using Boussinesq theory
/// 
/// The solitary wave has the form:
///   η(x,t) = a · sech²[k(x - ct - x₀)]
/// 
/// where:
/// - a = wave amplitude
/// - k = √(3a / 4h³) = characteristic width parameter
/// - c = √(g(h + a)) = wave celerity
/// - h = still water depth
/// - x₀ = initial wave crest position
pub struct SolitaryWaveGenerator {
    /// Wave amplitude a (m)
    pub amplitude: f64,
    /// Still water depth h (m)
    pub depth: f64,
    /// Gravitational acceleration (m/s²)
    pub gravity: f64,
    /// Wave celerity c (m/s)
    pub celerity: f64,
    /// Width parameter k (1/m)
    pub k: f64,
    /// Initial wave crest position x₀ (m)
    pub x0: f64,
    /// Generation position (left boundary)
    pub gen_x: f64,
}

impl SolitaryWaveGenerator {
    /// Create a new solitary wave generator from configuration
    pub fn new(config: &WaveConfig, gravity: f64) -> Self {
        let a = config.amplitude;
        let h = config.depth;
        
        // Wave celerity: c = sqrt(g(h + a))
        let c = (gravity * (h + a)).sqrt();
        
        // Width parameter: k = sqrt(3a / 4h³)
        let k = (3.0 * a / (4.0 * h.powi(3))).sqrt();
        
        // Initial crest position: start 3 wavelengths to the left of generation point
        // "Wavelength" of solitary wave ≈ 2π/k
        let wavelength_approx = 2.0 * PI / k;
        let x0 = config.generation_x - wavelength_approx;
        
        Self {
            amplitude: a,
            depth: h,
            gravity,
            celerity: c,
            k,
            x0,
            gen_x: config.generation_x,
        }
    }

    /// Calculate free surface elevation η at position x and time t
    /// η(x,t) = a · sech²[k(x - ct - x₀)]
    pub fn eta(&self, x: f64, t: f64) -> f64 {
        let xi = self.k * (x - self.celerity * t - self.x0);
        self.amplitude * (1.0 / xi.cosh()).powi(2)
    }

    /// Calculate horizontal velocity at position (x, y) and time t
    /// Using first-order Boussinesq theory:
    /// u(x,y,t) = c · η / (h + η)
    /// 
    /// For more accuracy with y-dependence:
    /// u = c·(η/h)·[1 - η/(4h) + h²/3·(1 - 3y²/h²)·(η/h)·k²]
    pub fn u_velocity(&self, x: f64, y: f64, t: f64) -> f64 {
        let eta = self.eta(x, t);
        let h = self.depth;
        
        // Simple depth-averaged velocity
        // This matches IH-2VOF's first-order approach
        self.celerity * eta / (h + eta)
    }

    /// Calculate vertical velocity at position (x, y) and time t
    /// v = -(y + h) · ∂u/∂x
    /// 
    /// Using sech² derivative: d/dx[sech²(kξ)] = -2k·sech²(kξ)·tanh(kξ)
    pub fn v_velocity(&self, x: f64, y: f64, t: f64) -> f64 {
        let xi = self.k * (x - self.celerity * t - self.x0);
        let sech2 = (1.0 / xi.cosh()).powi(2);
        let tanh_xi = xi.tanh();
        
        // dη/dx = -2ak·sech²(kξ)·tanh(kξ)
        let deta_dx = -2.0 * self.amplitude * self.k * sech2 * tanh_xi;
        
        let eta = self.amplitude * sech2;
        let h = self.depth;
        
        // du/dx from u = c·η/(h+η)
        let du_dx = self.celerity * h * deta_dx / (h + eta).powi(2);
        
        // v = -(y + h) · du/dx (assuming incompressibility)
        -(y + h) * du_dx
    }

    /// Water level (still water + wave) at position x and time t
    pub fn water_level(&self, x: f64, t: f64) -> f64 {
        self.depth + self.eta(x, t)
    }

    /// Apply solitary wave boundary condition at left boundary
    pub fn apply_boundary(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
        still_water_level: f64,
    ) {
        // Apply at left boundary (i = 0)
        let x_boundary = 0.0;
        let water_level = self.water_level(x_boundary, t);
        
        // Apply horizontal velocity at left boundary
        for j in 0..mesh.ny {
            let y_center = mesh.y_center(j);
            
            if y_center <= water_level {
                fields.u[[0, j]] = self.u_velocity(x_boundary, y_center, t);
            } else {
                fields.u[[0, j]] = 0.0;
            }
        }
        
        // Apply vertical velocity at left boundary
        for j in 0..=mesh.ny {
            let y_face = mesh.y_face(j);
            if y_face <= water_level {
                fields.v[[0, j]] = self.v_velocity(x_boundary, y_face, t);
            } else {
                fields.v[[0, j]] = 0.0;
            }
        }
        
        // Update VOF at left boundary
        for j in 0..mesh.ny {
            let y_bottom = mesh.y_face(j);
            let y_top = mesh.y_face(j + 1);
            
            if y_top <= water_level {
                fields.vof[[0, j]] = 1.0;
            } else if y_bottom >= water_level {
                fields.vof[[0, j]] = 0.0;
            } else {
                fields.vof[[0, j]] = (water_level - y_bottom) / mesh.dy;
            }
        }
    }

    /// Initialize the domain with the solitary wave
    /// Call this at t=0 to set initial conditions
    /// 
    /// Only sets the VOF field (free surface profile).
    /// Velocities start at zero and develop naturally.
    pub fn initialize_field(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
    ) {
        // Debug: log parameters
        log::info!("Initializing solitary wave: a={:.3}m, h={:.3}m, x0={:.2}m, k={:.3}/m", 
            self.amplitude, self.depth, self.x0, self.k);
        
        // Debug: check water level at key points
        let wl_peak = self.water_level(self.x0, t);
        let wl_x1 = self.water_level(1.0, t);
        let wl_x2 = self.water_level(2.0, t);
        log::info!("Water levels at t={:.2}: x0={:.4}m, x=1m={:.4}m, x=2m={:.4}m", 
            t, wl_peak, wl_x1, wl_x2);
        
        for i in 0..mesh.nx {
            let x = mesh.x_center(i);
            let water_level = self.water_level(x, t);
            
            // Set VOF only - velocities will develop from pressure gradients
            for j in 0..mesh.ny {
                // Skip solid cells (bathymetry)
                if fields.is_solid(i, j) {
                    continue;
                }
                
                let y_bottom = mesh.y_face(j);
                let y_top = mesh.y_face(j + 1);
                
                if y_top <= water_level {
                    fields.vof[[i, j]] = 1.0;
                } else if y_bottom >= water_level {
                    fields.vof[[i, j]] = 0.0;
                } else {
                    fields.vof[[i, j]] = (water_level - y_bottom) / mesh.dy;
                }
            }
        }
        
        // Velocities remain at zero - wave will propagate due to hydrostatic pressure gradient
        // This is similar to a "dam break" initialization where the elevated water column
        // creates horizontal pressure gradients that drive the flow.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WaveType;

    fn test_wave() -> WaveGenerator {
        let config = WaveConfig {
            enabled: true,
            wave_type: WaveType::Periodic,
            height: 0.1,
            amplitude: 0.0,
            period: 2.0,
            depth: 1.0,
            paddle_x0: 0.5,  // Paddle at 0.5m from left
            generation_x: 0.0,
        };
        WaveGenerator::new(&config, 9.81)
    }

    #[test]
    fn test_transfer_function() {
        let wave = test_wave();
        assert!(wave.stroke > 0.0);
        
        let kd = wave.k * wave.depth;
        let two_kd = 2.0 * kd;
        let transfer_ratio = 2.0 * (two_kd.cosh() - 1.0) / (two_kd.sinh() + two_kd);
        let expected_stroke = wave.height / transfer_ratio;
        
        assert!((wave.stroke - expected_stroke).abs() < 1e-10);
    }

    #[test]
    fn test_paddle_velocity() {
        let wave = test_wave();
        
        // At t=0: paddle velocity should be maximum
        let u = wave.paddle_velocity(0.0);
        let expected = 0.5 * wave.stroke * wave.omega;
        assert!((u - expected).abs() < 1e-10);
        
        // At t=T/4: paddle velocity should be zero
        let u = wave.paddle_velocity(wave.period / 4.0);
        assert!(u.abs() < 1e-10);
    }

    #[test]
    fn test_paddle_position() {
        let wave = test_wave();
        
        // At t=0: paddle at x0
        let x = wave.paddle_position(0.0);
        assert!((x - wave.x0).abs() < 1e-10);
        
        // At t=T/4: paddle at maximum
        let x = wave.paddle_position(wave.period / 4.0);
        let expected = wave.x0 + 0.5 * wave.stroke;
        assert!((x - expected).abs() < 1e-10);
    }

    #[test]
    fn test_is_solid() {
        let wave = test_wave();
        
        // At t=0, paddle is at x0=0.5
        // Points to the left should be solid
        assert!(wave.is_solid(0.3, 0.0));
        assert!(!wave.is_solid(0.6, 0.0));
    }

    #[test]
    fn test_wavelength() {
        let wave = test_wave();
        let l = wave.wavelength();
        assert!(l > 1.0);
        assert!(l < 10.0);
    }

    fn test_solitary() -> SolitaryWaveGenerator {
        let config = WaveConfig {
            enabled: true,
            wave_type: WaveType::Solitary,
            height: 0.0,
            amplitude: 0.07,
            period: 0.0,
            depth: 0.25,
            paddle_x0: 0.0,
            generation_x: 0.0,
        };
        SolitaryWaveGenerator::new(&config, 9.81)
    }

    #[test]
    fn test_solitary_celerity() {
        let wave = test_solitary();
        // c = sqrt(g(h + a)) = sqrt(9.81 * 0.32) ≈ 1.77 m/s
        let expected = (9.81 * 0.32_f64).sqrt();
        assert!((wave.celerity - expected).abs() < 0.01);
    }

    #[test]
    fn test_solitary_eta() {
        let wave = test_solitary();
        // At crest (x = ct + x0), η = a
        let t = 1.0;
        let x_crest = wave.celerity * t + wave.x0;
        let eta = wave.eta(x_crest, t);
        assert!((eta - wave.amplitude).abs() < 1e-10);
        
        // Far from crest, η → 0
        let eta_far = wave.eta(x_crest + 10.0, t);
        assert!(eta_far < 0.001);
    }

    #[test]
    fn test_solitary_velocity() {
        let wave = test_solitary();
        let t = 0.0;
        let x = wave.x0;  // at crest position
        
        // At crest, u = c * a / (h + a)
        let expected_u = wave.celerity * wave.amplitude / (wave.depth + wave.amplitude);
        let u = wave.u_velocity(x, 0.0, t);
        assert!((u - expected_u).abs() < 0.01);
        
        // Far from wave, u ≈ 0
        let u_far = wave.u_velocity(x + 10.0, 0.0, t);
        assert!(u_far.abs() < 0.01);
    }
}

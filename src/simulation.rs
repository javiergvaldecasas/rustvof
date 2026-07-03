//! Main simulation loop

use crate::config::{Config, WaveType, OutputFormat};
use crate::mesh::Mesh;
use crate::fields::Fields;
use crate::properties::FluidProperties;
use crate::solver::{compute_advection, compute_diffusion_variable, PressureSolver, project_velocity_variable_density, advect_vof, enforce_solid_bc, KEpsilonModel};
use crate::solver::projection::compute_intermediate_velocity;
use crate::boundary::{WaveGenerator, SolitaryWaveGenerator};
use crate::output::{VtkWriter, ProbeWriter};

#[cfg(feature = "netcdf-output")]
use crate::output::NetcdfWriter;


/// Enum to hold either wave generator type
pub enum WaveGen {
    Periodic(WaveGenerator),
    Solitary(SolitaryWaveGenerator),
}

impl WaveGen {
    /// Apply wave boundary condition
    pub fn apply_boundary(
        &self,
        fields: &mut Fields,
        mesh: &Mesh,
        t: f64,
        still_water_level: f64,
    ) {
        match self {
            WaveGen::Periodic(gen) => gen.apply_boundary(fields, mesh, t, still_water_level),
            WaveGen::Solitary(gen) => gen.apply_boundary(fields, mesh, t, still_water_level),
        }
    }
    
    /// Check if this is a moving paddle generator (periodic only)
    pub fn has_moving_paddle(&self) -> bool {
        match self {
            WaveGen::Periodic(gen) => gen.x0 > 0.0,
            WaveGen::Solitary(_) => false,
        }
    }
    
    /// Get paddle position (periodic only, returns 0 for solitary)
    pub fn paddle_position(&self, t: f64) -> f64 {
        match self {
            WaveGen::Periodic(gen) => gen.paddle_position(t),
            WaveGen::Solitary(_) => 0.0,
        }
    }
}

/// Main simulation structure
pub struct Simulation {
    /// Configuration
    pub config: Config,
    /// Mesh
    pub mesh: Mesh,
    /// Fields
    pub fields: Fields,
    /// Fluid properties
    pub props: FluidProperties,
    /// Current time
    pub time: f64,
    /// Current time step
    pub dt: f64,
    /// Time step number
    pub step: usize,
    /// Pressure solver (CPU iterative SOR)
    pressure_solver: PressureSolver,
    /// VTK writer
    vtk_writer: Option<VtkWriter>,
    /// NetCDF writer
    #[cfg(feature = "netcdf-output")]
    netcdf_writer: Option<NetcdfWriter>,
    /// Probe writer for time series
    probe_writer: ProbeWriter,
    /// Wave generator (optional)
    wave_gen: Option<WaveGen>,
    /// Time of last output
    last_output_time: f64,
    /// k-epsilon turbulence model (optional)
    turbulence_model: Option<KEpsilonModel>,
}

impl Simulation {
    /// Create a new simulation from configuration
    pub fn new(config: Config) -> Self {
        let mesh = Mesh::new(&config.domain);
        let mut fields = Fields::new(&mesh);
        let props = FluidProperties::from_config(&config.fluid);
        
        // Initialize fields
        fields.initialize(
            &mesh,
            &config.initial_condition,
            props.rho_water,
            props.rho_air,
            props.nu_water,
            props.nu_air,
        );
        
        // Apply bathymetry (mark solid cells based on bottom elevation)
        Self::apply_bathymetry(&mut fields, &mesh, &config);
        
        // Wave generator
        let wave_gen = if config.wave.enabled {
            match config.wave.wave_type {
                WaveType::Solitary => {
                    let gen = SolitaryWaveGenerator::new(&config.wave, props.gravity);
                    // Don't initialize field - generate wave at boundary instead
                    // This avoids initial transients from sudden pressure gradients
                    // gen.initialize_field(&mut fields, &mesh, 0.0);
                    log::info!("Solitary wave generator: a={:.3}m, c={:.2}m/s, k={:.2}/m (boundary-generated)",
                        gen.amplitude, gen.celerity, gen.k);
                    Some(WaveGen::Solitary(gen))
                }
                WaveType::Periodic => {
                    let gen = WaveGenerator::new(&config.wave, props.gravity);
                    log::info!("Periodic wave generator: H={:.3}m, T={:.2}s, L={:.2}m",
                        gen.height, gen.period, gen.wavelength());
                    Some(WaveGen::Periodic(gen))
                }
            }
        } else {
            None
        };
        
        // Initialize output writers based on format
        let vtk_writer = match config.output.format {
            OutputFormat::Vtk | OutputFormat::Both => Some(VtkWriter::new(&config.output.prefix)),
            OutputFormat::Netcdf => None,
        };
        
        #[cfg(feature = "netcdf-output")]
        let netcdf_writer = match config.output.format {
            OutputFormat::Netcdf | OutputFormat::Both => {
                Some(NetcdfWriter::new(&config.output.prefix, config.output.frames_per_file))
            }
            OutputFormat::Vtk => None,
        };
        
        let probe_writer = ProbeWriter::new(&config.probes, &mesh, &config.output.prefix);
        
        // Initialize pressure solver based on config
        let pressure_solver = PressureSolver {
            max_iter: config.solver.max_iter,
            tolerance: config.solver.tolerance,
            omega: config.solver.omega,
        };
        
        log::info!("Using iterative SOR pressure solver");

        // Initialize turbulence model if enabled
        let turbulence_model = if config.turbulence.enabled {
            log::info!("Turbulence model: k-epsilon enabled");
            log::info!("  Intensity: {:.1}%, Length scale: {:.3}m", 
                config.turbulence.intensity * 100.0, config.turbulence.length_scale);
            Some(KEpsilonModel::new())
        } else {
            None
        };
        
        let mut sim = Self {
            config,
            mesh,
            fields,
            props,
            time: 0.0,
            dt: 0.0,
            step: 0,
            pressure_solver,
            vtk_writer,
            #[cfg(feature = "netcdf-output")]
            netcdf_writer,
            probe_writer,
            wave_gen,
            last_output_time: -1.0, // Force output at t=0

            turbulence_model,
        };
        
        // Initialize turbulence fields if enabled
        if sim.config.turbulence.enabled {
            if let Some(ref model) = sim.turbulence_model {
                let (k_init, eps_init) = model.initialize(
                    &sim.fields,
                    &sim.mesh,
                    sim.config.turbulence.intensity,
                    sim.config.turbulence.length_scale,
                );
                sim.fields.k = k_init;
                sim.fields.epsilon = eps_init;
                sim.fields.nu_t = model.compute_nu_t(&sim.fields.k, &sim.fields.epsilon, &sim.fields.nu);
            }
        }
        
        sim
    }
    
    /// Apply bathymetry configuration to mark solid cells
    fn apply_bathymetry(fields: &mut Fields, mesh: &Mesh, config: &Config) {
        use crate::config::BathymetryType;
        
        if config.bathymetry.bathy_type == BathymetryType::Flat {
            return; // Nothing to do
        }
        
        log::info!("Applying bathymetry: {:?}", config.bathymetry.bathy_type);
        
        // Mark cells as solid based on bathymetry
        for i in 0..mesh.nx {
            let x = mesh.x_center(i);
            let bottom = config.bathymetry.bottom_at(x);
            
            for j in 0..mesh.ny {
                let y_center = mesh.y_center(j);
                
                if y_center < bottom {
                    // Mark as solid
                    fields.set_solid(i, j);
                    fields.ac[[i, j]] = 0.0;  // For visualization
                    fields.vof[[i, j]] = 1.0; //0.0;
                    fields.u[[i, j]] = 0.0;
                    fields.v[[i, j]] = 0.0;
                }
            }
        }
        
        // Zero velocities at solid boundaries
        for i in 0..mesh.nx {
            for j in 0..mesh.ny {
                if fields.is_solid(i, j) {
                    // Zero u-faces adjacent to this solid cell
                    fields.u[[i, j]] = 0.0;
                    if i + 1 <= mesh.nx { fields.u[[i + 1, j]] = 0.0; }
                    // Zero v-faces adjacent to this solid cell
                    fields.v[[i, j]] = 0.0;
                    if j + 1 <= mesh.ny { fields.v[[i, j + 1]] = 0.0; }
                }
            }
        }
        
        // Log statistics
        let solid_cells = fields.cell_type.iter().filter(|&&ct| ct.is_solid()).count();
        log::info!("Bathymetry: {} solid cells", solid_cells);
    }
    

    /// Run the simulation
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        log::info!("Starting simulation");
        log::info!("Domain: {}x{} cells ({:.2}m x {:.2}m)", 
            self.mesh.nx, self.mesh.ny, self.mesh.length_x, self.mesh.length_y);
        log::info!("End time: {:.2}s", self.config.time.t_end);
        

        log::info!("GPU acceleration: DISABLED (using CPU)");


        #[cfg(not(feature = "gpu"))]
        log::info!("GPU acceleration: NOT COMPILED (cpu-only build)");
        
        // Initial output
        self.write_output()?;
        
        // Initial probe recording
        self.probe_writer.record(
            self.time,
            &self.mesh,
            &self.fields,
            self.config.time.dt_output,
        );
        
        // Main time loop
        while self.time < self.config.time.t_end {
            // Compute adaptive time step
            self.compute_dt();
            
            // Advance one time step
            self.step()?;
            
            // Update time
            self.time += self.dt;
            self.step += 1;
            
            // Output if needed
            if self.time - self.last_output_time >= self.config.time.dt_output {
                self.write_output()?;
            }
            
            // Record probe data
            self.probe_writer.record(
                self.time,
                &self.mesh,
                &self.fields,
                self.config.time.dt_output,
            );
            
            // Progress
            if self.step % 100 == 0 {
                let div = self.fields.total_divergence(&self.mesh);
                let vol = self.fields.total_water_volume(&self.mesh);
                log::info!(
                    "Step {}: t={:.4}s, dt={:.6}s, div={:.2e}, vol={:.4}",
                    self.step, self.time, self.dt, div, vol
                );
            }
        }
        
        // Final output
        self.write_output()?;
        
        // Final probe recording and flush
        self.probe_writer.record(
            self.time,
            &self.mesh,
            &self.fields,
            self.config.time.dt_output,
        );
        self.probe_writer.flush();
        
        log::info!("Simulation complete at t={:.4}s ({} steps)", self.time, self.step);
        
        Ok(())
    }

    /// Compute adaptive time step based on CFL condition
    fn compute_dt(&mut self) {
        let max_vel = self.fields.max_velocity().max(0.01); // Avoid division by zero
        let dt_cfl = self.config.time.cfl * self.mesh.dx.min(self.mesh.dy) / max_vel;
        
        // Viscous stability limit (include turbulent viscosity if enabled)
        let mut nu_max = self.props.nu_water.max(self.props.nu_air);
        if self.turbulence_model.is_some() {
            // Find max effective viscosity
            let nu_t_max = self.fields.nu_t.iter().cloned().fold(0.0_f64, f64::max);
            nu_max = (nu_max + nu_t_max).max(nu_max);
        }
        let dt_visc = 0.25 * self.mesh.dx.min(self.mesh.dy).powi(2) / nu_max;
        
        // Take minimum
        self.dt = dt_cfl.min(dt_visc);
        
        // Apply maximum dt if specified
        if let Some(dt_max) = self.config.time.dt_max {
            self.dt = self.dt.min(dt_max);
        }
        
        // Don't overshoot end time
        if self.time + self.dt > self.config.time.t_end {
            self.dt = self.config.time.t_end - self.time;
        }
    }

    /// Advance one time step
    /// Advance one time step (simplified, follows IH2VOF structure)
    /// 
    /// Order of operations (Chorin projection method):
    /// 1. BC + wave generation (once at start)
    /// 2. Update cell types (moving paddle)
    /// 3. Compute u* = uⁿ + Δt(advection + diffusion + gravity)
    /// 4. Solve pressure Poisson equation
    /// 5. Project velocity: uⁿ⁺¹ = u* - Δt∇p/ρ
    /// 6. Advect VOF
    /// 7. Apply final BC + update properties
    fn step(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let has_waves = self.wave_gen.is_some();
        let t_new = self.time + self.dt;

        // === 1. Boundary conditions (once at start of step) ===
        self.apply_walls_bc(has_waves);
        
        if let Some(ref wave_gen) = self.wave_gen {
            wave_gen.apply_boundary(
                &mut self.fields, 
                &self.mesh, 
                self.time,
                self.config.initial_condition.water_level,
            );
        }
        
        // === 2. Update cell types for moving paddle ===
        if let Some(ref wave_gen) = self.wave_gen {
            if wave_gen.has_moving_paddle() {
                let x_paddle = wave_gen.paddle_position(t_new);
                self.fields.set_solid_behind_paddle(&self.mesh, x_paddle);
            }
        }
        
        // === 3. Compute intermediate velocity u* ===
        // Ensure solid boundaries have zero velocity before computing advection
        self.zero_solid_velocity();
        
        // If turbulence is enabled, add nu_t to effective viscosity
        if self.turbulence_model.is_some() {
            // Add turbulent viscosity: nu_eff = nu_laminar + nu_t
            for j in 0..self.mesh.ny {
                for i in 0..self.mesh.nx {
                    self.fields.nu[[i, j]] += self.fields.nu_t[[i, j]];
                }
            }
        }
        
        // u* = uⁿ + Δt(-advection + diffusion + gravity)
        let (adv_u, adv_v) = compute_advection(&self.fields, &self.mesh);
        let (diff_u, diff_v) = compute_diffusion_variable(&self.fields, &self.mesh);
        
        compute_intermediate_velocity(
            &mut self.fields,
            &adv_u, &adv_v,
            &diff_u, &diff_v,
            &self.mesh,
            self.dt,
            self.props.gravity,
        );
        
        // Zero solid velocity after intermediate step
        self.zero_solid_velocity();
        
        // Apply BC to u* (like IH2VOF does at end of VTILDE)
        self.apply_walls_bc(has_waves);
        
        // Re-apply wave paddle BC after intermediate velocity calculation
        // This is crucial - the paddle velocity must be imposed as a Dirichlet BC
        if let Some(ref wave_gen) = self.wave_gen {
            wave_gen.apply_boundary(
                &mut self.fields, 
                &self.mesh, 
                self.time,
                self.config.initial_condition.water_level,
            );
        }
        
        // === 4. Solve pressure Poisson equation ===
        let rhs = PressureSolver::compute_rhs(&self.fields, &self.mesh, self.dt);
        
        // Choose solver based on config
        let (converged, iters, residual) = {

            {
                self.solve_pressure_cpu(&rhs)

            }
            #[cfg(not(feature = "gpu"))]
            {
                self.solve_pressure_cpu(&rhs)
            }
        };
        
        if !converged {
            log::warn!("Pressure solver did not converge: {} iters, residual={:.2e}", 
                iters, residual);
        }
        
        // === 5. Project velocity ===
        // uⁿ⁺¹ = u* - Δt·∇p/ρ
        project_velocity_variable_density(&mut self.fields, &self.mesh, self.dt);
        
        // === 6. Advect VOF ===
        advect_vof(&mut self.fields, &self.mesh, self.dt);
        
        // === 7. Final BC + update properties ===
        self.apply_walls_bc(has_waves);
        self.zero_solid_velocity();
        
        // Update fluid properties (resets nu to laminar values)
        self.fields.update_properties(
            self.props.rho_water,
            self.props.rho_air,
            self.props.nu_water,
            self.props.nu_air,
        );
        
        // === 8. Advance turbulence (if enabled) ===
        if let Some(ref model) = self.turbulence_model {
            // Advance k and epsilon
            let (k_new, eps_new) = model.advance(
                &self.fields,
                &self.fields.k,
                &self.fields.epsilon,
                &self.fields.nu_t,
                &self.mesh,
                self.dt,
            );
            self.fields.k = k_new;
            self.fields.epsilon = eps_new;
            
            // Update turbulent viscosity for next step
            self.fields.nu_t = model.compute_nu_t(
                &self.fields.k, 
                &self.fields.epsilon, 
                &self.fields.nu,
            );
        }
        
        Ok(())
    }

    /// Write output files
    fn write_output(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Write VTK if enabled
        if let Some(ref mut writer) = self.vtk_writer {
            writer.write(&self.mesh, &self.fields, self.time)?;
        }
        
        // Write NetCDF if enabled
        #[cfg(feature = "netcdf-output")]
        if let Some(ref mut writer) = self.netcdf_writer {
            writer.write(&self.mesh, &self.fields, self.time)
                .map_err(|e| format!("NetCDF write error: {}", e))?;
        }
        
        self.last_output_time = self.time;
        Ok(())
    }

    /// Get water volume (for conservation check)
    pub fn water_volume(&self) -> f64 {
        self.fields.total_water_volume(&self.mesh)
    }

    /// Get velocity divergence (for incompressibility check)
    pub fn divergence(&self) -> f64 {
        self.fields.total_divergence(&self.mesh)
    }

    /// Apply wall boundary conditions
    /// If skip_left is true, don't apply BC on left wall (for wave generation)
    fn apply_walls_bc(&mut self, skip_left: bool) {
        // Right wall: solid wall (u = 0) - allows wave reflection
        // For absorption, a sponge layer would be needed (not implemented yet)
        for j in 0..self.mesh.ny {
            self.fields.u[[self.mesh.nx, j]] = 0.0;
        }
        
        // Left wall: u = 0 (only if no wave generation)
        if !skip_left {
            for j in 0..self.mesh.ny {
                self.fields.u[[0, j]] = 0.0;
            }
        }
        
        // Bottom wall: v = 0
        for i in 0..self.mesh.nx {
            self.fields.v[[i, 0]] = 0.0;
        }
        
        // Top boundary: v = 0 (closed tank)
        for i in 0..self.mesh.nx {
            self.fields.v[[i, self.mesh.ny]] = 0.0;
        }
    }

    /// Zero velocity in solid cells (using cell_type field)
    fn zero_solid_velocity(&mut self) {
        // Zero u-velocity at faces adjacent to PERMANENT solid cells (bathymetry)
        // Don't zero at Paddle faces - the wave generator sets velocity there
        for j in 0..self.mesh.ny {
            for i in 0..=self.mesh.nx {
                // u is on vertical faces - check adjacent cells
                // Only zero if adjacent to permanent solid (bathymetry), not Paddle
                let left_perm_solid = i > 0 && self.fields.cell_type[[i - 1, j]].is_permanent_solid();
                let right_perm_solid = i < self.mesh.nx && self.fields.cell_type[[i, j]].is_permanent_solid();
                
                if left_perm_solid || right_perm_solid {
                    self.fields.u[[i, j]] = 0.0;
                }
            }
        }
        
        // Zero v-velocity at faces adjacent to PERMANENT solid cells
        for j in 0..=self.mesh.ny {
            for i in 0..self.mesh.nx {
                // v is on horizontal faces - check adjacent cells
                let below_perm_solid = j > 0 && self.fields.cell_type[[i, j - 1]].is_permanent_solid();
                let above_perm_solid = j < self.mesh.ny && self.fields.cell_type[[i, j]].is_permanent_solid();
                
                if below_perm_solid || above_perm_solid {
                    self.fields.v[[i, j]] = 0.0;
                }
            }
        }
        
        // Zero velocity INSIDE Paddle cells (not at their boundary with fluid)
        for j in 0..self.mesh.ny {
            for i in 0..self.mesh.nx {
                if self.fields.cell_type[[i, j]].is_paddle() {
                    self.fields.u[[i, j]] = 0.0;
                    self.fields.v[[i, j]] = 0.0;
                }
            }
        }
    }

    /// Solve pressure equation on CPU (iterative SOR)
    fn solve_pressure_cpu(&mut self, rhs: &ndarray::Array2<f64>) -> (bool, usize, f64) {
        self.pressure_solver.solve_with_fields(&mut self.fields, rhs, &self.mesh)
    }
}

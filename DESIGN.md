# VOF2D - Technical Design

## 1. Overview

2D VOF solver based on the **projection method** (Chorin, 1968) for the
incompressible Navier-Stokes equations with free-surface tracking via Volume of
Fluid.

### Main Features

- **Two-phase** water-air flow with variable density and viscosity
- **Staggered mesh** of MAC type (Marker-And-Cell)
- **k-ε turbulence model** (optional)
- **SOR pressure solver** with variable density
- **Variable bathymetry** (flat or linear slope)
- **Wave generation** with a moving paddle (piston wavemaker) or solitary waves
- **VOF advection** with a TVD Van Leer scheme (2nd order)
- **Output** in VTK and/or NetCDF

> **Note:** GPU acceleration is *not* implemented. The `wgpu`/`bytemuck` dependencies
> and the `gpu` feature flag exist as scaffolding only; the `--gpu` CLI flag is a
> no-op. All computation runs on the CPU (with `rayon` available for parallelism).

### Main Algorithm (per timestep)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Compute adaptive Δt (CFL + diffusion + turbulence)                  │
│  2. Apply wall boundary conditions                                      │
│  3. [If waves] Apply wave generator (with gain ramp)                    │
│  4. [If moving paddle] Update the solid-cell mask                       │
│  5. [If bathymetry] Mark solid cells below the bottom                   │
│  6. Compute advective terms: (u·∇)u                                     │
│  7. Compute diffusive terms: (ν + νt)∇²u                               │
│  8. Compute intermediate velocity: u* = uⁿ + Δt(-adv + diff - F·g)      │
│  9. Solve the Poisson equation: ∇·(1/ρ ∇p) = ∇·u*/Δt  [SOR]             │
│ 10. Correct velocity: uⁿ⁺¹ = u* - Δt·∇p/ρ                               │
│ 11. Apply boundary conditions                                           │
│ 12. Advect the VOF field: Fⁿ⁺¹ = Fⁿ - Δt·∇·(Fu) (TVD scheme)            │
│ 13. [If k-ε] Advect and solve k and ε, update νt                        │
│ 14. Update properties (ρ, ν_eff) from VOF and turbulence                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Staggered Mesh (MAC Grid)

**Marker-And-Cell** mesh (Harlow & Welch, 1965):

```
      j+1  ●───v───●───v───●───v───●
           │       │       │       │
           u  P,F  u  P,F  u  P,F  u
           │  ρ,ν  │  ρ,ν  │  ρ,ν  │
       j   ●───v───●───v───●───v───●
           │       │       │       │
           u  P,F  u  P,F  u  P,F  u
           │  ρ,ν  │  ρ,ν  │  ρ,ν  │
      j-1  ●───v───●───v───●───v───●
          i-1     i      i+1     i+2

    ● = mesh vertices
    P, F, ρ, ν = cell centers [i,j]
    u = vertical faces [i,j] where i ∈ [0, nx]
    v = horizontal faces [i,j] where j ∈ [0, ny]
```

### Array dimensions

| Field | Size | Location |
|-------|------|----------|
| VOF (F) | nx × ny | Cell centers |
| Pressure (p) | nx × ny | Cell centers |
| Density (ρ) | nx × ny | Cell centers |
| Viscosity (ν) | nx × ny | Cell centers |
| CellType | nx × ny | Cell centers |
| Velocity u | (nx+1) × ny | Vertical faces |
| Velocity v | nx × (ny+1) | Horizontal faces |

### Coordinates

```rust
// Cell center [i,j]
x_center = (i + 0.5) * dx
y_center = (j + 0.5) * dy

// Vertical face (u) [i,j]
x_face = i * dx
y_face = (j + 0.5) * dy

// Horizontal face (v) [i,j]
x_face = (i + 0.5) * dx
y_face = j * dy
```

---

## 3. Governing Equations

### 3.1 Incompressible Navier-Stokes (continuous form)

**Momentum conservation:**
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + g
```

**Incompressibility:**
```
∇·u = 0
```

**VOF advection:**
```
∂F/∂t + ∇·(Fu) = 0
```

### 3.2 Two-phase fluid properties

```
ρ(F) = F·ρ_water + (1-F)·ρ_air
ν(F) = F·ν_water + (1-F)·ν_air
```

| Property | Water | Air |
|----------|-------|-----|
| ρ (kg/m³) | 1000 | 1.225 |
| ν (m²/s) | 1.0×10⁻⁶ | 1.5×10⁻⁵ |

---

## 4. Projection Method (Chorin)

### 4.1 Step 1: Intermediate Velocity

We compute u* without considering pressure:

```
u* = uⁿ + Δt·[-(u·∇)u + ν∇²u]
v* = vⁿ + Δt·[-(u·∇)v + ν∇²v - F_face·g]
```

### 4.2 Gravity Treatment (VOF-weighted)

**As implemented**, the gravity source term applied to the vertical velocity is
weighted by the VOF fraction interpolated onto the v-face, so gravity acts on
water but not on air:

```rust
// In projection.rs: compute_intermediate_velocity()

// VOF interpolated onto the v-face (average of the two adjacent cells)
let vof_face = 0.5 * (vof_below + vof_above);

// Gravity weighted by VOF:
//   - In water (F ≈ 1): full gravity g
//   - In air   (F ≈ 0): no gravity
let g_effective = gravity * vof_face;

v*[i,j] += dt * (-adv_v + diff_v - g_effective);
```

**Notes / limitations:**
- This is the simple VOF-weighted approach. It is known to create small
  artificial gradients at the interface (the pressure gradient does not exactly
  balance the discontinuous gravity term), which can generate weak spurious
  velocities near the free surface.
- A reduced-density buoyancy formulation (`g·(ρ − ρ_air)/ρ`, using
  `fields.rho_at_v`) was considered as a mitigation but is **not** the code path
  currently in use — `compute_intermediate_velocity()` uses the VOF-weighted term
  above.

### 4.3 Step 2: Pressure Poisson Equation

To make u divergence-free, we solve:

```
∇·(1/ρ ∇p) = ∇·u*/Δt
```

**Discretization (5-point stencil with variable density):**

```
Σ (1/ρ_face) · (p_neighbor - p_ij) / h² = div(u*) / Δt
```

where the density on each face is obtained from the **arithmetic mean of the VOF**
of the two adjacent cells (not a harmonic mean of the densities):
```
ρ_face = get_rho( ½·(F[i,j] + F[neighbor]) )
   with get_rho(F) = F·ρ_water + (1-F)·ρ_air
```

### 4.4 Step 3: Velocity Correction

```
uⁿ⁺¹ = u* - Δt·(∂p/∂x)/ρ_face
vⁿ⁺¹ = v* - Δt·(∂p/∂y)/ρ_face
```

**Implementation:**
```rust
// In projection.rs: project_velocity_variable_density()
let dp_dx = (p[i,j] - p[i-1,j]) / dx;
let rho_face = fields.rho_at_u(i, j, mesh);
u[i,j] -= dt * dp_dx / rho_face;
```

---

## 5. Pressure Solver (SOR with VOF)

### 5.1 Algorithm

We use **Successive Over-Relaxation (SOR)** with ω = 1.5:

```
p_new = (sum - rhs) / coeff
p[i,j] = p_old + ω·(p_new - p_old)
```

### 5.2 Boundary Conditions

| Boundary | Condition | Implementation |
|----------|-----------|----------------|
| Walls (left, right, bottom) | Neumann: ∂p/∂n = 0 | p_ghost = p_interior |
| Free surface (VOF < 0.1) | Dirichlet: p = 0 | p = 0 (atmospheric) |
| Top of the domain | Dirichlet: p = 0 | p = 0 at j = ny-1 |
| Solid cells | Excluded | p = 0, does not take part in the solve |

### 5.3 VOF interface treatment

```rust
// In pressure.rs: solve_with_vof()
for each cell [i,j]:
    // Solid cells: excluded
    if fields.is_solid(i, j):
        continue

    // Air: atmospheric pressure
    if vof[i,j] < 0.1:
        p[i,j] = 0
        continue

    // For each neighbor:
    if vof[neighbor] < 0.1:
        p_neighbor = 0  // Dirichlet at the interface
    else:
        p_neighbor = p[neighbor]

    // Accumulate with weight 1/ρ, where ρ_face = get_rho(½·(F_c + F_neighbor))
    sum += (1/ρ_face) * p_neighbor / h²
    coeff += (1/ρ_face) / h²
```

> **Note:** In the code, the actual per-step solve is `solve_with_fields()` (invoked
> via `Simulation::solve_pressure_cpu`), which implements exactly this VOF-aware,
> variable-density SOR sweep. The sibling methods `solve()` (constant-coefficient)
> and `solve_with_vof()` exist but are not on the main timestep path.

### 5.4 Solver parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_iter | 10000 | Maximum iterations |
| tolerance | 1×10⁻⁶ | Convergence tolerance |
| omega | 1.5 | SOR over-relaxation factor |

---

## 5B. k-ε Turbulence Model

### 5B.1 Equations

The standard k-ε model (Launder & Spalding, 1974) adds two transport equations:

**Turbulent kinetic energy (k):**
```
∂k/∂t + u·∇k = ∇·[(ν + νt/σk)∇k] + P - ε
```

**Dissipation rate (ε):**
```
∂ε/∂t + u·∇ε = ∇·[(ν + νt/σε)∇ε] + C1ε·(ε/k)·P - C2ε·ε²/k
```

**Turbulent viscosity:**
```
νt = Cμ · k² / ε
```

**Turbulence production:**
```
P = νt · |S|²
```
where |S| is the magnitude of the strain-rate tensor:
```
|S|² = 2·Sij·Sij,  Sij = ½(∂ui/∂xj + ∂uj/∂xi)
```

### 5B.2 Model constants

| Constant | Value | Description |
|----------|-------|-------------|
| Cμ | 0.09 | Turbulent viscosity coefficient |
| C1ε | 1.44 | ε production coefficient |
| C2ε | 1.92 | ε dissipation coefficient |
| σk | 1.0 | Turbulent Prandtl number for k |
| σε | 1.3 | Turbulent Prandtl number for ε |

### 5B.3 Initial conditions

From turbulence intensity (I) and length scale (l):

```
k₀ = 1.5 · (I · U)²
ε₀ = Cμ^0.75 · k₀^1.5 / l
```

**Configuration:**
```toml
[turbulence]
enabled = true
model = "k-epsilon"
intensity = 0.05      # 5% turbulence intensity
length_scale = 0.05   # 5cm length scale
```

### 5B.4 Implementation

```rust
// In solver/turbulence.rs
pub struct KEpsilonModel {
    pub constants: KEpsilonConstants,
    pub k_min: f64,      // 1e-10 (avoids div/0)
    pub eps_min: f64,    // 1e-10
    pub nu_t_ratio_max: f64,  // 1e5 (limits νt/ν)
}

impl KEpsilonModel {
    /// Magnitude of the strain-rate tensor |S|
    pub fn strain_rate_magnitude(&self, fields: &Fields, mesh: &Mesh,
                                  i: usize, j: usize) -> f64 {
        let du_dx = (fields.u[[i+1, j]] - fields.u[[i, j]]) / mesh.dx;
        let dv_dy = (fields.v[[i, j+1]] - fields.v[[i, j]]) / mesh.dy;
        let du_dy = /* centered interpolation */;
        let dv_dx = /* centered interpolation */;

        let s11 = du_dx;
        let s22 = dv_dy;
        let s12 = 0.5 * (du_dy + dv_dx);

        (2.0 * (s11*s11 + s22*s22 + 2.0*s12*s12)).sqrt()
    }

    /// Turbulent viscosity νt = Cμ·k²/ε
    pub fn turbulent_viscosity(&self, k: f64, eps: f64, nu: f64) -> f64 {
        let k_safe = k.max(self.k_min);
        let eps_safe = eps.max(self.eps_min);
        let nu_t = self.constants.c_mu * k_safe * k_safe / eps_safe;
        nu_t.min(self.nu_t_ratio_max * nu)
    }

    /// Advance k and ε by one time step
    pub fn advance(&self, fields: &mut Fields, mesh: &Mesh, dt: f64) {
        // 1. Compute production P = νt·|S|²
        // 2. Advect k and ε (upwind)
        // 3. Diffuse k and ε
        // 4. Add source/sink terms
        // 5. Update νt over the whole domain
    }
}
```

### 5B.5 VOF interface treatment

Turbulence is applied mainly in the water phase:

```rust
// Effective viscosity taking VOF into account
let nu_eff = if vof > 0.5 {
    nu_water + nu_t
} else {
    nu_air  // No turbulence in air
};
```

### 5B.6 Wall functions

For cells adjacent to solids, logarithmic wall functions are applied:

```
u⁺ = (1/κ)·ln(y⁺) + B

where:
  u⁺ = u/u_τ
  y⁺ = y·u_τ/ν
  u_τ = √(τ_w/ρ)
  κ = 0.41 (von Kármán constant)
  B = 5.0
```

### 5B.7 Limitations of k-ε for waves

The standard k-ε model has limitations for free-surface flows:

1. **Overproduction of νt** near the interface
2. **Does not capture wave breaking** (requires additional models)
3. **Excessive damping** of short waves

Possible improvements:
- k-ω SST model (better near walls)
- Durbin limiter for stagnation regions
- Turbulence damping at the free surface

---

## 6. VOF Advection

### 6.1 TVD Scheme with Van Leer Limiter

We use a second-order TVD (Total Variation Diminishing) scheme with a Van Leer
limiter to reduce numerical diffusion while avoiding oscillations.

```
Fⁿ⁺¹[i,j] = Fⁿ[i,j] - Δt·(flux_x/dx + flux_y/dy)
```

**Face value with TVD:**
```
F_face = F_upwind + 0.5 · ψ(r) · (F_downwind - F_upwind)
```

**Van Leer limiter:**
```
ψ(r) = (r + |r|) / (1 + |r|)

where r = (F_center - F_upwind) / (F_downwind - F_center)
```

The limiter:
- r > 0 (smooth gradient): ψ → second order
- r ≤ 0 (local extremum): ψ = 0 → first order (stable)

### 6.2 Clamping

After advection, we constrain F ∈ [0, 1]:
```rust
fields.clamp_vof();  // F = max(0, min(1, F))
```

---

## 7. Advective Terms

### 7.1 For velocity u

```
(u·∇)u = ∂(u²)/∂x + ∂(uv)/∂y
```

**Upwind discretization:**
```rust
// d(u²)/dx
if u[i,j] >= 0:
    du2_dx = (u[i,j]² - u[i-1,j]²) / dx
else:
    du2_dx = (u[i+1,j]² - u[i,j]²) / dx

// d(uv)/dy - requires interpolating v to the u location
v_at_u = average of the 4 neighboring v
```

### 7.2 For velocity v

```
(u·∇)v = ∂(uv)/∂x + ∂(v²)/∂y
```

Analogous, interpolating u to the v location.

### 7.3 Solid-cell treatment

Solid cells (e.g. behind the paddle) are excluded from the advection
computation:

```rust
let left_solid = fields.is_solid(i - 1, j);
let right_solid = fields.is_solid(i, j);
if left_solid || right_solid {
    continue;  // adv = 0
}
```

---

## 8. Diffusive Terms

### 8.1 Laplacian

```
ν∇²u = ν·(∂²u/∂x² + ∂²u/∂y²)
```

**Central discretization:**
```
∂²u/∂x² ≈ (u[i+1,j] - 2·u[i,j] + u[i-1,j]) / dx²
∂²u/∂y² ≈ (u[i,j+1] - 2·u[i,j] + u[i,j-1]) / dy²
```

### 8.2 Variable viscosity

The viscosity is interpolated onto the faces using VOF:

```rust
// In fields.rs: nu_at_u(), nu_at_v()
let vof_face = 0.5 * (vof[i-1,j] + vof[i,j]);
let nu_face = vof_face * nu_water + (1.0 - vof_face) * nu_air;
```

---

## 9. Boundary Conditions

### 9.1 Solid Walls

```rust
// Left wall: u = 0
u[0, j] = 0  for all j

// Right wall: u = 0 (wave reflection)
u[nx, j] = 0  for all j

// Bottom: v = 0
v[i, 0] = 0  for all i

// Top: v = 0 (closed tank) or free
v[i, ny] = 0  for all i
```

### 9.2 Wall types

```rust
pub enum WallType {
    NoSlip,    // u = 0 at the wall
    FreeSlip,  // u·n = 0, ∂u_t/∂n = 0
}
```

---

## 10. Wave Generation

### 10.1 Piston Wavemaker (Moving Paddle)

The paddle oscillates inside the domain, creating a solid region behind it:

```
┌─────────────────────────────────────────────────────────────┐
│  WALL  │ SOLID      │ PADDLE│      FLUID (waves →)    │     │
│        │ (blocked)  │      │                          │     │
├────────┼────────────┼──────┼──────────────────────────┼─────┤
│  x=0   │            │ x_paddle(t)                     │ x=L │
└─────────────────────────────────────────────────────────────┘
```

**Piston transfer function:**
```
H/S = 2(cosh(2kd) - 1) / (sinh(2kd) + 2kd)
```

where:
- H = wave height
- S = piston stroke
- k = wave number
- d = depth

### 10.2 Gain Ramp (Ramp-up)

To avoid abrupt transients, the amplitude increases gradually over the first
2 periods:

```rust
// In waves.rs: WaveGenerator
pub ramp_periods: f64,  // Default: 2.0

/// Smooth (cosine) gain factor
pub fn ramp_gain(&self, t: f64) -> f64 {
    let t_ramp = self.ramp_periods * self.period;

    if t >= t_ramp {
        1.0  // Full amplitude
    } else if t <= 0.0 {
        0.0
    } else {
        // Cosine ramp: 0.5 * (1 - cos(π * t / T_ramp))
        0.5 * (1.0 - (PI * t / t_ramp).cos())
    }
}
```

**Behavior:**
| Time | Gain |
|------|------|
| t = 0 | 0% |
| t = T | 50% |
| t = 2T | 100% |

**Paddle position and velocity:**
```rust
pub fn paddle_position(&self, t: f64) -> f64 {
    let gain = self.ramp_gain(t);
    self.x0 + gain * 0.5 * self.stroke * (self.omega * t).sin()
}

pub fn paddle_velocity(&self, t: f64) -> f64 {
    let gain = self.ramp_gain(t);
    let gain_derivative = ...;  // Derivative of the ramp

    // Full derivative: d/dt[gain(t) * (S/2) * sin(ωt)]
    gain_derivative * 0.5 * self.stroke * sin_wt
        + gain * 0.5 * self.stroke * self.omega * cos_wt
}
```

### 10.3 Solitary Waves (Boussinesq)

For solitary waves:

```
η(x,t) = a · sech²[k(x - ct - x₀)]
```

where:
- a = amplitude
- c = √(g(h + a)) = celerity
- k = √(3a / 4h³) = width parameter
- h = depth

---

## 10B. Variable Bathymetry

### 10B.1 Bottom types

```rust
pub enum BathymetryType {
    Flat,    // Flat bottom (y = 0)
    Slope,   // Linear slope
    Custom,  // From file (not implemented)
}
```

### 10B.2 Linear slope (Slope)

Defines a ramp between two points:

```
┌─────────────────────────────────────────────────────────────┐
│                                              ░░░░░░░░░░░░░░░│
│  WATER                                  ░░░░░░░░░░░░░░░░░░░░│
│                                    ░░░░░░░░░░░░░░░░░░░░░░░░░│
│                               ░░░░░░░░░░░░░ BEACH ░░░░░░░░░░│
│                          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
├─────────────────────░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░┤
│        FLAT BOTTOM  │      SLOPE          │  CONSTANT       │
│  y = 0              │  y = f(x)           │  y = end_y      │
├─────────────────────┼─────────────────────┼─────────────────┤
│  x < start_x        │  start_x ≤ x ≤ end_x│  x > end_x      │
└─────────────────────────────────────────────────────────────┘
```

**Bottom-elevation function:**

```rust
pub fn bottom_at(&self, x: f64) -> f64 {
    match self.bathy_type {
        BathymetryType::Flat => 0.0,
        BathymetryType::Slope => {
            if x < self.slope_start_x {
                0.0  // Flat before
            } else if x >= self.slope_end_x {
                self.slope_end_y  // Constant after
            } else {
                // Linear interpolation
                let t = (x - self.slope_start_x) /
                        (self.slope_end_x - self.slope_start_x);
                self.slope_start_y + t * (self.slope_end_y - self.slope_start_y)
            }
        }
        BathymetryType::Custom => unimplemented!(),
    }
}
```

### 10B.3 Configuration

```toml
[bathymetry]
type = "slope"
slope_start_x = 4.0   # Slope start (m)
slope_start_y = 0.0   # Initial elevation (m)
slope_end_x = 14.0    # Slope end (m)
slope_end_y = 0.5     # Final elevation (m)
```

**Example: Beach with a 5% slope:**
- Slope = (0.5 - 0.0) / (14.0 - 4.0) = 0.05 = 5%
- Effective depth at x=14m: d(x) = water_level - bottom(x)

### 10B.4 Implementation in the solver

Cells below the bathymetry are marked as **solid**:

```rust
// In simulation.rs: update every step
for i in 0..mesh.nx {
    let x = mesh.x_center(i);
    let bottom = config.bathymetry.bottom_at(x);

    for j in 0..mesh.ny {
        let y = mesh.y_center(j);
        if y < bottom {
            fields.cell_type[[i, j]] = CellType::Solid;
        }
    }
}
```

### 10B.5 Physical effects

Bathymetry produces:

1. **Shoaling**: Increase in wave height as depth decreases
   ```
   H₂/H₁ = √(Cg₁/Cg₂)
   ```

2. **Refraction**: Change of direction (in 3D)

3. **Breaking**: When H/d > ~0.8 (McCowan criterion)

4. **Run-up**: Water rising up the slope

---

## 11. Cell Classification

```rust
#[repr(u8)]
pub enum CellType {
    Fluid  = 0,  // Water or air (VOF determines the fraction)
    Solid  = 1,  // Permanent blocked cell (walls, bathymetry)
    Porous = 2,  // Partial permeability (Darcy/Forchheimer) — reserved, not yet used
    Paddle = 3,  // Temporary solid behind the wave paddle (resets as it moves)
}
```

`is_solid()` returns true for both `Solid` and `Paddle`; `is_permanent_solid()`
returns true only for `Solid`. A companion `PorousProperties` struct is defined for
future porous-media support but is not exercised by the solver yet.

### 11.1 Fluid Cells (Fluid)
- Normal behavior: advection, diffusion, pressure
- VOF ∈ [0,1] determines water vs air

### 11.2 Solid / Paddle Cells
- Velocity = 0 always
- Excluded from the pressure solver (skipped in the SOR sweep)
- No-penetration condition on adjacent faces
- `Paddle` cells are updated dynamically (moving paddle); `Solid` cells (walls,
  bathymetry) are permanent

```rust
// In fields.rs
pub fn set_solid_behind_paddle(&mut self, mesh: &Mesh, x_paddle: f64) {
    for i in 0..mesh.nx {
        let x = mesh.x_center(i);
        for j in 0..mesh.ny {
            if x < x_paddle {
                self.cell_type[[i, j]] = CellType::Solid;
            } else {
                self.cell_type[[i, j]] = CellType::Fluid;
            }
        }
    }
}
```

---

## 12. Numerical Stability

### 12.1 CFL Condition

```
Δt_CFL ≤ CFL · min(Δx, Δy) / max(|u|, |v|)
```

With a typical CFL = 0.3 - 0.5

### 12.2 Diffusion Condition

```
Δt_visc ≤ 0.25 · min(Δx², Δy²) / ν_max
```

### 12.3 Adaptive Timestep

```rust
fn compute_dt(&mut self) {
    let max_vel = self.fields.max_velocity().max(0.01);
    let dt_cfl = self.config.time.cfl * self.mesh.dx.min(self.mesh.dy) / max_vel;

    let nu_max = self.props.nu_water.max(self.props.nu_air);
    let dt_visc = 0.25 * self.mesh.dx.min(self.mesh.dy).powi(2) / nu_max;

    self.dt = dt_cfl.min(dt_visc);

    if let Some(dt_max) = self.config.time.dt_max {
        self.dt = self.dt.min(dt_max);
    }
}
```

---

## 13. Code Structure

```
src/
├── main.rs              # CLI (only the `run <config.toml> [--gpu]` subcommand)
├── lib.rs               # Public re-exports
├── config.rs            # TOML parsing, config structures (incl. bathymetry, turbulence)
├── mesh.rs              # Staggered mesh, coordinates, indices
├── fields.rs            # Field arrays (VOF, u, v, p, ρ, ν, k, ε, νt, cell_type)
├── cell_type.rs         # CellType enum (Fluid, Solid)
├── properties.rs        # Property interpolation
├── simulation.rs        # Main time loop
│
├── solver/
│   ├── mod.rs           # Solver re-exports
│   ├── advection.rs     # Computation of (u·∇)u with upwind
│   ├── diffusion.rs     # Computation of ν∇²u
│   ├── pressure.rs      # SOR Poisson solver with VOF
│   ├── projection.rs    # Intermediate velocity + correction + buoyancy
│   ├── turbulence.rs    # k-ε model (equations, production, νt)
│   └── vof.rs           # VOF advection with TVD Van Leer
│
├── boundary/
│   ├── mod.rs
│   ├── walls.rs         # Solid-wall conditions
│   └── waves.rs         # Wave generator (piston + solitary + ramp)
│
├── output/
│   ├── mod.rs
│   ├── vtk.rs           # VTK Legacy format writing
│   ├── netcdf.rs        # NetCDF-3 writing (CF-compliant, behind `netcdf-output`)
│   └── probes.rs        # Point time series
```

> There is **no** `src/gpu/` module. GPU acceleration described in earlier drafts of
> this document was never implemented; the `gpu` Cargo feature is an empty flag.

---

## 14. Configuration (TOML)

### 14.1 Full structure

```toml
[solver]
pressure_solver = "iterative"  # iterative SOR solver
max_iter = 1000                # Maximum iterations
tolerance = 1e-4               # Convergence tolerance

[domain]
length_x = 4.0    # Domain length (m)
length_y = 0.8    # Domain height (m)
nx = 200          # Cells in x
ny = 40           # Cells in y

[time]
t_end = 15.0      # End time (s)
dt_output = 0.05  # Output interval (s)
cfl = 0.4         # CFL number
dt_max = 0.002    # Maximum Δt (s)

[fluid]
rho_water = 1000.0
rho_air = 1.225
nu_water = 1.0e-6
nu_air = 1.5e-5
gravity = 9.81

[turbulence]
enabled = true            # Enable the turbulence model
model = "k-epsilon"       # Model: "k-epsilon" (the only one for now)
intensity = 0.05          # Initial turbulence intensity (5%)
length_scale = 0.05       # Turbulent length scale (m)

[initial_condition]
water_level = 0.5  # Initial level (m)

[wave]
enabled = true
type = "periodic"   # "periodic" or "solitary"
height = 0.05       # Wave height H (m) - for periodic
amplitude = 0.07    # Amplitude a (m) - for solitary
period = 1.5        # Period T (s)
depth = 0.5         # Depth d (m)
paddle_x0 = 0.3     # Initial paddle position (m), 0 = fixed boundary

[bathymetry]
type = "flat"       # "flat", "slope"
slope_start_x = 5.0
slope_start_y = 0.0
slope_end_x = 10.0
slope_end_y = 0.3

[output]
format = "netcdf"       # "vtk", "netcdf", "both"
prefix = "output/sim"
frames_per_file = 50    # Frames per NetCDF file (0 = a single file)

[probes]
enabled = true
file = "probes.csv"
dt = 0.01               # Sampling interval
wave_gauges = [1.0, 2.0, 3.0]  # x positions of the gauges
```

---

## 15. NetCDF Output

### 15.1 Enabling

```bash
cargo build --release --features netcdf-output
```

Or in Cargo.toml:
```toml
[features]
default = ["gpu", "netcdf-output"]
```

### 15.2 File structure

**Dimensions:**
- `x` (nx): x coordinates
- `y` (ny): y coordinates
- `time` (unlimited): time

**Variables:**

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `x` | (x) | m | x coordinate |
| `y` | (y) | m | y coordinate |
| `time` | (time) | s | Time |
| `vof` | (time, y, x) | 1 | Volume fraction |
| `p` | (time, y, x) | Pa | Pressure |
| `rho` | (time, y, x) | kg/m³ | Density |
| `u` | (time, y, x) | m/s | x velocity (at center) |
| `v` | (time, y, x) | m/s | y velocity (at center) |
| `eta` | (time, x) | m | Free-surface elevation |

### 15.3 Block writing

For long simulations, output can be split into multiple files:

```toml
[output]
frames_per_file = 50  # Generates sim_000.nc, sim_001.nc, ...
```

Each file contains 50 frames. The existing file is deleted before writing.

### 15.4 Loading with xarray

```python
import xarray as xr

# One file
ds = xr.open_dataset("output/sim.nc")

# Multiple files
ds = xr.open_mfdataset("output/sim_*.nc", combine='nested', concat_dim='time')
```

---

## 16. Analysis Tools (Python)

**Not present in this repository.** No `analysis/` package (`VOFDataset`, plotter,
animator, `vof_viz.py`) ships with the code. NetCDF output is standard CF-compliant
NetCDF-3, so it can be loaded directly with `xarray`:

```python
import xarray as xr
ds = xr.open_dataset("output/sim.nc")           # single file
ds = xr.open_mfdataset("output/sim_*.nc",        # multiple files
                       combine='nested', concat_dim='time')
```

---

## 17. GPU Acceleration

**Not implemented.** This section documented a planned `wgpu`-based GPU backend that
does not exist in the codebase:

- There is no `src/gpu/` module, no `GpuContext`, and no `Simulation::enable_gpu()`.
- The `gpu` Cargo feature is an empty flag; `wgpu`/`pollster`/`bytemuck` are declared
  as dependencies but unused.
- The `--gpu` CLI flag is accepted but ignored (no-op).

All solvers (pressure SOR, advection, diffusion, projection, VOF, k-ε) run on the
CPU.

---

## 18. Verification and Validation

### 18.1 Mass Conservation

There is **no** dedicated `test-sloshing` CLI subcommand (the only subcommand is
`run`). A ready-made sloshing configuration is available programmatically via
`Config::sloshing_tank()`, and mass conservation / divergence can be monitored from
the periodic log line emitted every 100 steps:

```
Step N: t=..., dt=..., div=<total divergence>, vol=<total water volume>
```

Expected behaviour: total water volume stays essentially constant and the total
divergence remains near machine-small (~10⁻⁷).

### 18.2 Hydrostatic Equilibrium

A column of water at rest should stay approximately static (∂p/∂y = ρg, v ≈ 0).
Note that with the VOF-weighted gravity term (see §4.2) small spurious velocities
may appear at the free surface.

---

## 19. Known Limitations

1. **No surface tension**: No capillary effects
2. **k-ε limited for breaking**: The model does not capture wave breaking well
3. **No absorption**: Waves reflect off all walls (no sponge layers)
4. **2D only**: No 3D extension
5. **CPU only**: No GPU acceleration is implemented (see §17)
6. **No PLIC**: Algebraic VOF, no geometric interface reconstruction
7. **VOF-weighted gravity**: The simple `F·g` source term can produce weak spurious
   velocities at the interface (see §4.2)

---

## 20. References

### Fundamental numerical methods
1. Chorin, A.J. (1968). "Numerical solution of the Navier-Stokes equations"
2. Harlow, F.H., Welch, J.E. (1965). "Numerical calculation of time-dependent viscous incompressible flow"
3. Hirt, C.W., Nichols, B.D. (1981). "Volume of fluid (VOF) method for the dynamics of free boundaries"
4. Ferziger, J.H., Perić, M. (2002). "Computational Methods for Fluid Dynamics"

### k-ε model
5. Launder, B.E., Spalding, D.B. (1974). "The numerical computation of turbulent flows"
6. Wilcox, D.C. (2006). "Turbulence Modeling for CFD"
7. Pope, S.B. (2000). "Turbulent Flows" — Ch. 10-11

### Waves and breaking
8. IH-2VOF User's Manual, IHCantabria
9. Bradford, S.F. (2000). "Numerical simulation of surf zone dynamics"
10. Jacobsen, N.G. et al. (2012). "A wave generation toolbox for the open-source CFD library: OpenFoam"
11. Higuera, P. et al. (2013). "Realistic wave generation and active wave absorption for Navier-Stokes models"
12. McCowan, J. (1894). "On the highest wave of permanent type" — Breaking criterion H/d = 0.78

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-02-24 | Initial design |
| 2026-02-24 | Fix: VOF-weighted gravity |
| 2026-02-24 | Fix: pressure solver with variable density |
| 2026-02-24 | Improvement: VOF advection with TVD Van Leer scheme |
| 2026-02-24 | Feature: moving-paddle generator |
| 2026-02-25 | Feature: GPU acceleration with wgpu |
| 2026-02-26 | Feature: NetCDF-3 output (CF-compliant) |
| 2026-02-27 | Feature: gain ramp (2 periods) on the paddle |
| 2026-02-27 | Fix: buoyancy with reduced density (removes leaks) |
| 2026-02-27 | Feature: frames_per_file for NetCDF |
| 2026-02-27 | Feature: Python analysis tools (VOFDataset) |
| 2026-02-27 | Fix: delete existing NetCDF file before writing |
| 2026-03-02 | Feature: **k-ε turbulence model** complete |
| 2026-03-02 | Feature: **Variable bathymetry** (linear slope for beaches) |
| 2026-03-02 | Feature: Dynamic solid cells below bathymetry |
| 2026-03-02 | Improvement: Effective viscosity ν_eff = ν + νt with turbulence |
| 2026-03-03 | Doc: Full update of DESIGN.md with new features |
| 2026-07-01 | Removed the FFT pressure solver; SOR is the only pressure solver |
| 2026-07-01 | Doc: Translated documentation to English |
| 2026-07-03 | Doc: Synced DESIGN.md with the actual code — gravity is VOF-weighted (not reduced-density buoyancy); pressure face density uses arithmetic VOF mean (not harmonic); GPU acceleration and the Python analysis tools are documented as not implemented; CellType has 4 variants (Fluid/Solid/Porous/Paddle); removed the non-existent `test-sloshing` command |
```


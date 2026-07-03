# rustvof

A two-dimensional **VOF (Volume of Fluid)** solver written in Rust for the
simulation of free-surface, two-phase (water-air) flow, aimed at wave
propagation in channels (wave flumes).

> ## ⚠️ Work in progress — NOT validated
>
> This code is **under active development** and **has not been validated**
> against experimental data or reference solvers. The results **must not be
> used for engineering or research purposes** without independent verification.
> The API, configuration formats and numerical schemes may change without
> notice.

## Features

- 2D staggered Cartesian mesh (staggered MAC grid).
- Explicit incompressible Navier-Stokes.
- VOF advection for tracking the water-air interface.
- SOR pressure solver with variable density.
- Regular wave generation (Airy theory, piston-type wavemaker).
- Solid-wall boundary conditions (no-slip / reflection).
- Output in VTK (ParaView) and NetCDF-3.


## Requirements

- Rust (2021 edition or newer).

## Building

```bash
cargo build --release
```

To build without GPU support:

```bash
cargo build --release --no-default-features --features netcdf-output
```

## Usage

Run a simulation from a TOML configuration file:

```bash
cargo run --release -- run examples/paddle_slope.toml
```


## Layout

```
src/        Solver code (mesh, fields, solver, gpu, boundary, output)
examples/   TOML configurations and benchmarks
```

## License

MIT — see [`LICENSE`](LICENSE).

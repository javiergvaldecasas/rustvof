//! 2D Staggered mesh (MAC grid) implementation
//!
//! Uses a Marker-and-Cell (MAC) arrangement where:
//! - Pressure and VOF are at cell centers
//! - u-velocity is at vertical cell faces (i+1/2, j)
//! - v-velocity is at horizontal cell faces (i, j+1/2)

use crate::config::DomainConfig;

/// 2D Staggered mesh
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Number of cells in x direction
    pub nx: usize,
    /// Number of cells in y direction
    pub ny: usize,
    /// Cell size in x direction (meters)
    pub dx: f64,
    /// Cell size in y direction (meters)
    pub dy: f64,
    /// Domain length in x (meters)
    pub length_x: f64,
    /// Domain length in y (meters)
    pub length_y: f64,
}

impl Mesh {
    /// Create a new mesh from domain configuration
    pub fn new(config: &DomainConfig) -> Self {
        Self {
            nx: config.nx,
            ny: config.ny,
            dx: config.dx(),
            dy: config.dy(),
            length_x: config.length_x,
            length_y: config.length_y,
        }
    }

    /// Total number of cells
    pub fn num_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Number of u-velocity points (on vertical faces)
    pub fn num_u(&self) -> (usize, usize) {
        (self.nx + 1, self.ny)
    }

    /// Number of v-velocity points (on horizontal faces)
    pub fn num_v(&self) -> (usize, usize) {
        (self.nx, self.ny + 1)
    }

    /// X coordinate of cell center
    pub fn x_center(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.dx
    }

    /// Y coordinate of cell center
    pub fn y_center(&self, j: usize) -> f64 {
        (j as f64 + 0.5) * self.dy
    }

    /// X coordinate of u-velocity point (vertical face)
    pub fn x_face(&self, i: usize) -> f64 {
        i as f64 * self.dx
    }

    /// Y coordinate of v-velocity point (horizontal face)
    pub fn y_face(&self, j: usize) -> f64 {
        j as f64 * self.dy
    }

    /// Get cell indices from physical coordinates
    /// Returns (i, j) of the cell containing point (x, y)
    pub fn cell_at(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        if x < 0.0 || x >= self.length_x || y < 0.0 || y >= self.length_y {
            return None;
        }
        let i = (x / self.dx) as usize;
        let j = (y / self.dy) as usize;
        Some((i.min(self.nx - 1), j.min(self.ny - 1)))
    }

    /// Linear index for cell (i, j) in row-major order
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Cell indices from linear index
    #[inline]
    pub fn cell_ij(&self, idx: usize) -> (usize, usize) {
        (idx % self.nx, idx / self.nx)
    }

    /// Check if cell (i, j) is on the boundary
    pub fn is_boundary(&self, i: usize, j: usize) -> bool {
        i == 0 || i == self.nx - 1 || j == 0 || j == self.ny - 1
    }

    /// Get boundary type for cell (i, j)
    pub fn boundary_type(&self, i: usize, j: usize) -> BoundaryType {
        let left = i == 0;
        let right = i == self.nx - 1;
        let bottom = j == 0;
        let top = j == self.ny - 1;

        match (left, right, bottom, top) {
            (true, _, true, _) => BoundaryType::CornerBottomLeft,
            (true, _, _, true) => BoundaryType::CornerTopLeft,
            (_, true, true, _) => BoundaryType::CornerBottomRight,
            (_, true, _, true) => BoundaryType::CornerTopRight,
            (true, _, _, _) => BoundaryType::Left,
            (_, true, _, _) => BoundaryType::Right,
            (_, _, true, _) => BoundaryType::Bottom,
            (_, _, _, true) => BoundaryType::Top,
            _ => BoundaryType::Interior,
        }
    }

    /// Iterator over all cell indices (i, j)
    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.ny).flat_map(move |j| (0..self.nx).map(move |i| (i, j)))
    }

    /// Iterator over interior cells (excluding boundary)
    pub fn interior_cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (1..self.ny - 1).flat_map(move |j| (1..self.nx - 1).map(move |i| (i, j)))
    }

    /// Iterator over u-velocity indices
    pub fn u_points(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (nu, nv) = self.num_u();
        (0..nv).flat_map(move |j| (0..nu).map(move |i| (i, j)))
    }

    /// Iterator over v-velocity indices
    pub fn v_points(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (nu, nv) = self.num_v();
        (0..nv).flat_map(move |j| (0..nu).map(move |i| (i, j)))
    }
}

/// Boundary type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    Interior,
    Left,
    Right,
    Bottom,
    Top,
    CornerBottomLeft,
    CornerBottomRight,
    CornerTopLeft,
    CornerTopRight,
}

impl BoundaryType {
    pub fn is_interior(&self) -> bool {
        matches!(self, BoundaryType::Interior)
    }

    pub fn is_boundary(&self) -> bool {
        !self.is_interior()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mesh() -> Mesh {
        Mesh {
            nx: 10,
            ny: 5,
            dx: 0.1,
            dy: 0.1,
            length_x: 1.0,
            length_y: 0.5,
        }
    }

    #[test]
    fn test_cell_coordinates() {
        let mesh = test_mesh();
        
        // First cell center
        assert!((mesh.x_center(0) - 0.05).abs() < 1e-10);
        assert!((mesh.y_center(0) - 0.05).abs() < 1e-10);
        
        // Last cell center
        assert!((mesh.x_center(9) - 0.95).abs() < 1e-10);
        assert!((mesh.y_center(4) - 0.45).abs() < 1e-10);
    }

    #[test]
    fn test_face_coordinates() {
        let mesh = test_mesh();
        
        // First face
        assert!((mesh.x_face(0)).abs() < 1e-10);
        assert!((mesh.y_face(0)).abs() < 1e-10);
        
        // Last face
        assert!((mesh.x_face(10) - 1.0).abs() < 1e-10);
        assert!((mesh.y_face(5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cell_at() {
        let mesh = test_mesh();
        
        assert_eq!(mesh.cell_at(0.05, 0.05), Some((0, 0)));
        assert_eq!(mesh.cell_at(0.95, 0.45), Some((9, 4)));
        assert_eq!(mesh.cell_at(-0.1, 0.0), None);
        assert_eq!(mesh.cell_at(1.1, 0.0), None);
    }

    #[test]
    fn test_boundary_type() {
        let mesh = test_mesh();
        
        assert_eq!(mesh.boundary_type(0, 0), BoundaryType::CornerBottomLeft);
        assert_eq!(mesh.boundary_type(9, 4), BoundaryType::CornerTopRight);
        assert_eq!(mesh.boundary_type(5, 0), BoundaryType::Bottom);
        assert_eq!(mesh.boundary_type(5, 4), BoundaryType::Top);
        assert_eq!(mesh.boundary_type(0, 2), BoundaryType::Left);
        assert_eq!(mesh.boundary_type(9, 2), BoundaryType::Right);
        assert_eq!(mesh.boundary_type(5, 2), BoundaryType::Interior);
    }

    #[test]
    fn test_num_points() {
        let mesh = test_mesh();
        
        assert_eq!(mesh.num_cells(), 50);
        assert_eq!(mesh.num_u(), (11, 5)); // nx+1 x ny
        assert_eq!(mesh.num_v(), (10, 6)); // nx x ny+1
    }
}

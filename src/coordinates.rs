use crate::{Float, MAX, MIN};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridCoordinate = usize;

/// Trait used to convert from SpaceCoordinate to GridCoordinate.
///
/// This is needed because one cannot write impl blocks on built-in primitives.
pub trait FromSpaceCoordinate {
    fn from_space_coordinate<const N_GRID: usize>(coord: SpaceCoordinate) -> GridCoordinate;
}

impl FromSpaceCoordinate for GridCoordinate {
    fn from_space_coordinate<const N_GRID: usize>(coord: SpaceCoordinate) -> GridCoordinate {
        let scaling: Float = N_GRID as Float / (MAX - MIN);
        ((coord * scaling).floor() - (MIN * scaling).floor()) as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_grid_coordinate_from_space_coordinate() {
        const N_GRID: usize = 511;
        let grid_coord = GridCoordinate::from_space_coordinate::<N_GRID>(MIN);
        assert_eq!(0, grid_coord);

        let grid_coord = GridCoordinate::from_space_coordinate::<N_GRID>(dbg!(MAX.next_down()));
        assert_eq!(N_GRID - 1, grid_coord);
    }
}

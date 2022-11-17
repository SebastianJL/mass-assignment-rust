use crate::{Float, MAX, MIN, N_GRID};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridCoordinate = usize;

/// Trait used to convert from SpaceCoordinate to GridCoordinate.
/// 
/// This is needed because one cannot write impl blocks on built-in primitives.
pub trait FromSpaceCoordinate {
    fn from_space_coordinate(coord: SpaceCoordinate) -> GridCoordinate;
}

impl FromSpaceCoordinate for GridCoordinate {
    fn from_space_coordinate(coord: SpaceCoordinate) -> GridCoordinate {
        const SCALING: Float = N_GRID as Float / (MAX - MIN);
        ((coord * SCALING).floor() - (MIN * SCALING).floor()) as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_grid_coordinate_from_space_coordinate() {
        let grid_coord = GridCoordinate::from_space_coordinate(MIN);
        assert_eq!(0, grid_coord);

        let grid_coord = GridCoordinate::from_space_coordinate(dbg!(MAX.next_down()));
        assert_eq!(N_GRID - 1, grid_coord);
    }
}

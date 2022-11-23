use crate::{Float, MAX, MIN};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridCoordinate = usize;

/// Return nearest grid point to `coord` of grid with size `N_GRID`.
///
/// Global constants [MIN], [MAX] are the extremes of the spacial coordinates.
/// # WARNING
/// This doesn't guarantee that the `result < N_GRID` for `coord <` [MAX].
/// So if you need that you need to take the appropriate measures on the result.
pub fn grid_coordinate<const N_GRID: usize>(coord: SpaceCoordinate) -> GridCoordinate {
    assert_ne!(N_GRID, 0_usize, "N_GRID cannot be 0.");
    let scaling = N_GRID as Float / (MAX - MIN);
    ((coord - MIN) * scaling).floor() as GridCoordinate
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_grid_coordinate<const N_GRID: usize>() {
        let grid_coord = grid_coordinate::<N_GRID>(MIN);
        assert_eq!(0, grid_coord);

        let grid_coord = grid_coordinate::<N_GRID>((MAX + MIN) / 2.);
        assert_eq!(N_GRID / 2, grid_coord);

        // Don't assert this edge case. It has to be handled differently depending on problem.
        // let grid_coord = grid_coordinate::<N_GRID>(MAX.next_down());
        // assert_eq!(N_GRID - 1, grid_coord);

        let grid_coord = grid_coordinate::<N_GRID>(MAX);
        assert_eq!(N_GRID, grid_coord);
    }

    #[test]
    #[should_panic(expected = "N_GRID cannot be 0.")]
    fn test_grid_coordinate_n_grid_is_000() {
        test_grid_coordinate::<000>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_001() {
        test_grid_coordinate::<001>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_002() {
        test_grid_coordinate::<002>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_006() {
        test_grid_coordinate::<006>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_127() {
        test_grid_coordinate::<127>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_128() {
        test_grid_coordinate::<128>();
    }

    #[test]
    fn test_grid_coordinate_n_grid_is_129() {
        test_grid_coordinate::<129>();
    }
}

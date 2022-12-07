use std::iter::repeat;

use crate::{Float, MAX, MIN};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridIndex = usize;
/// 1D discrete coordinate of hunk (multiple slabs of grid).
pub type HunkIndex = usize;

/// Return nearest grid point to `coord` of grid with size `N_GRID`.
///
/// Global constants [MIN], [MAX] are the extremes of the spacial coordinates.
/// # WARNING
/// This doesn't guarantee that the `result < N_GRID` for `coord <` [MAX].
/// So if you need that you need to take the appropriate measures on the result.
pub fn grid_index_from_coordinate<const N_GRID: usize>(coord: SpaceCoordinate) -> GridIndex {
    assert_ne!(N_GRID, 0_usize, "N_GRID cannot be 0.");
    let scaling = N_GRID as Float / (MAX - MIN);
    ((coord - MIN) * scaling).floor() as GridIndex
}

/// Distribute `n_grid` cells uniformly along `n_hunk` hunks.
///
/// Give all hunks `N_GRID / N_HUNK` cells. Give the first `N_GRID % N_HUNK` hunks one more.
/// Also included the N_GRID for convenience.
pub fn hunk_starting_indices_vec<const N_GRID: usize, const N_HUNK: usize>() -> Vec<GridIndex> {
    let rest = N_GRID % N_HUNK;
    let base_size = N_GRID / N_HUNK;
    let large_hunks = (0..rest).map(|hunk_index| (base_size + 1) * hunk_index);
    let small_hunks =
        (rest..=N_HUNK).map(|hunk_index| (base_size + 1) * rest + base_size * (hunk_index - rest));
    large_hunks.chain(small_hunks).collect()
}

pub fn grid_to_hunk_index_vec<const N_GRID: usize, const N_HUNK: usize>() -> Vec<HunkIndex> {
    let hunk_start = hunk_starting_indices_vec::<N_GRID, N_HUNK>();
    let mut result = Vec::with_capacity(N_GRID);
    for (i, w) in hunk_start.windows(2).into_iter().enumerate() {
        let hunk_width = w[1] - w[0];
        result.extend(repeat(i).take(hunk_width));
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_grid_coordinate<const N_GRID: usize>() {
        let grid_coord = grid_index_from_coordinate::<N_GRID>(MIN);
        assert_eq!(0, grid_coord);

        let grid_coord = grid_index_from_coordinate::<N_GRID>((MAX + MIN) / 2.);
        assert_eq!(N_GRID / 2, grid_coord);

        // Don't assert this edge case. It has to be handled differently depending on problem.
        // let grid_coord = grid_coordinate::<N_GRID>(MAX.next_down());
        // assert_eq!(N_GRID - 1, grid_coord);

        let grid_coord = grid_index_from_coordinate::<N_GRID>(MAX);
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

    #[test]
    fn test_grid_index_from_hunk() {
        const N_GRID: usize = 11;
        const N_HUNK: usize = 3;
        let hunk_starting_indices = hunk_starting_indices_vec::<N_GRID, N_HUNK>();
        assert_eq!(vec![0, 4, 8, 11], hunk_starting_indices);
    }

    #[test]
    fn test_grid_to_hunk_index() {
        const N_GRID: usize = 11;
        const N_HUNK: usize = 3;
        let grid_to_hunk = grid_to_hunk_index_vec::<N_GRID, N_HUNK>();
        assert_eq!(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], grid_to_hunk)
    }
}

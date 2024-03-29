use crate::{Float, MAX, MIN};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridIndex = usize;
/// 1D discrete coordinate of hunk (multiple pencils of grid).
pub type HunkIndex = usize;

/// Return nearest grid point to `coord` of grid with size `N_GRID`.
///
/// Global constants [MIN], [MAX] are the extremes of the spacial coordinates.
/// # WARNING
/// This doesn't guarantee that the `result < N_GRID` for `coord <` [MAX].
/// So if you need that you need to take the appropriate measures on the result.
///
/// # Panics
/// Panics if `N_GRID==0`.
pub fn grid_index_from_coordinate<const N_GRID: usize>(coord: SpaceCoordinate) -> GridIndex {
    assert_ne!(N_GRID, 0_usize, "N_GRID cannot be 0.");
    let scaling = N_GRID as Float / (MAX - MIN);
    ((coord - MIN) * scaling).floor() as GridIndex
}

pub const fn hunk_index_from_grid_index<const N_GRID: usize, const N_HUNKS: usize>(
    grid_index: GridIndex,
) -> HunkIndex {
    grid_index / get_hunk_size(N_GRID, N_HUNKS)
}

pub const fn get_hunk_size(n_grid: usize, n_hunks: usize) -> usize {
    (n_grid + n_hunks - 1) / n_hunks
}

pub const fn get_chunk_size(n_particles: usize, n_chunks: usize) -> usize {
    (n_particles + n_chunks - 1) / n_chunks
}

// pub fn

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
    fn test_hunk_index_from_grid_index() {
        const N_GRID: usize = 11;
        const N_HUNK: usize = 3;
        let hunk_starting_indices: Vec<_> = (0..N_GRID)
            .map(|i| hunk_index_from_grid_index::<N_GRID, N_HUNK>(i))
            .collect();
        assert_eq!(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], hunk_starting_indices);
    }

    #[test]
    fn test_hunk_size() {
        const N_GRID: usize = 11;
        const N_HUNK: usize = 3;

        assert_eq!(4, get_hunk_size(N_GRID, N_HUNK));
    }
}

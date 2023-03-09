use crate::{Float, MAX, MIN};

/// 1D continuous coordinate.
pub type SpaceCoordinate = Float;
/// 1D discrete coordinate on grid.
pub type GridIndex = usize;
/// 1D discrete coordinate of hunk (multiple slabs of grid).
pub type HunkIndex = usize;
/// 1D discrete coordinate of super cell.
pub type CellIndex = usize;

/// Return nearest grid point to `coord` of grid with size `N_GRID`.
///
/// Global constants [MIN], [MAX] are the extremes of the spacial coordinates.
///
/// # WARNING
/// This doesn't guarantee that the `result < N_GRID` for `coord <` [MAX].
/// So if you need that you need to take the appropriate measures on the result.
///
/// # Panics
/// Panics if `N_GRID==0`.
pub fn grid_index_from_coordinate(coord: SpaceCoordinate, n_grid: usize) -> GridIndex {
    assert_ne!(n_grid, 0_usize, "N_GRID cannot be 0.");
    let scaling = n_grid as Float / (MAX - MIN);
    ((coord - MIN) * scaling).floor() as GridIndex
}

pub const fn hunk_index_from_grid_index(
    grid_index: GridIndex,
    n_grid: usize,
    n_hunks: usize,
) -> HunkIndex {
    grid_index / get_hunk_size(n_grid, n_hunks)
}

pub const fn get_hunk_size(n_grid: usize, n_hunks: usize) -> usize {
    (n_grid + n_hunks - 1) / n_hunks
}

pub const fn get_chunk_size(n_particles: usize, n_chunks: usize) -> usize {
    (n_particles + n_chunks - 1) / n_chunks
}

pub const fn get_cell_size() -> usize {
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_grid_coordinate<const N_GRID: usize>() {
        let grid_coord = grid_index_from_coordinate(MIN, N_GRID);
        assert_eq!(0, grid_coord);

        let grid_coord = grid_index_from_coordinate((MAX + MIN) / 2., N_GRID);
        assert_eq!(N_GRID / 2, grid_coord);

        // Don't assert this edge case. It has to be handled differently depending on problem.
        // let grid_coord = grid_coordinate::<N_GRID>(MAX.next_down());
        // assert_eq!(N_GRID - 1, grid_coord);

        let grid_coord = grid_index_from_coordinate(MAX, N_GRID);
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
            .map(|i| hunk_index_from_grid_index(i, N_GRID, N_HUNK))
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

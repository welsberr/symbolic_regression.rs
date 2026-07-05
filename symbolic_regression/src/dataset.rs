use dynamic_expressions::utils::ZipEq;
use fastrand::Rng;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::random::usize_range;

#[derive(Copy, Clone, Debug)]
pub struct TaggedDataset<'a, T: Float> {
    pub data: &'a Dataset<T>,
    pub baseline_loss: Option<T>,
    /// Row count of the *full* dataset this tag refers to. Equal to `data.n_rows` when the tag
    /// wraps the full dataset; when it wraps a batch buffer, this is the underlying full size.
    /// Mirrors SymbolicRegression.jl's `SubDataset` / `BasicDataset` distinction.
    pub full_n_rows: usize,
}

impl<'a, T: Float> TaggedDataset<'a, T> {
    pub fn new(data: &'a Dataset<T>, baseline_loss: Option<T>) -> Self {
        Self {
            data,
            baseline_loss,
            full_n_rows: data.n_rows,
        }
    }
    pub fn for_batch(batch: &'a Dataset<T>, baseline_loss: Option<T>, full_n_rows: usize) -> Self {
        Self {
            data: batch,
            baseline_loss,
            full_n_rows,
        }
    }
    /// Matches SymbolicRegression.jl `dataset_fraction`: ratio of `data.n_rows` to the full size.
    pub fn dataset_fraction(&self) -> f64 {
        if self.full_n_rows == 0 {
            1.0
        } else {
            (self.data.n_rows as f64) / (self.full_n_rows as f64)
        }
    }
}

impl<'a, T: Float> std::ops::Deref for TaggedDataset<'a, T> {
    type Target = Dataset<T>;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

#[derive(Clone, Debug)]
pub struct Dataset<T: Float> {
    /// Column-major contiguous data with shape `(n_features, n_rows)` for vectorization over rows.
    pub x: Array2<T>,
    /// Target vector with length `n_rows`.
    pub y: Array1<T>,
    pub n_features: usize,
    pub n_rows: usize,
    pub weights: Option<Array1<T>>,
    pub variable_names: Vec<String>,
    /// Optional sequence/group id per row. Delays are valid only within a sequence.
    pub sequence_ids: Option<Vec<usize>>,
    /// Optional inclusive lower target bound per row.
    pub target_low: Option<Array1<T>>,
    /// Optional inclusive upper target bound per row.
    pub target_high: Option<Array1<T>>,
    /// Weighted mean of `y` (or unweighted mean when no weights).
    pub avg_y: T,
}

impl<T: Float> Dataset<T> {
    fn build_dataset(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        sequence_ids: Option<Vec<usize>>,
        target_low: Option<Array1<T>>,
        target_high: Option<Array1<T>>,
        avg_y: Option<T>,
    ) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_features, n_rows) = x.dim();
        assert_eq!(y.len(), n_rows);
        if let Some(ref w) = weights {
            assert_eq!(w.len(), n_rows);
        }
        if let Some(ref ids) = sequence_ids {
            assert_eq!(ids.len(), n_rows);
            assert!(
                ids.windows(2).all(|w| w[0] <= w[1]),
                "sequence_ids must be nondecreasing so each sequence is contiguous"
            );
        }
        match (&target_low, &target_high) {
            (Some(low), Some(high)) => {
                assert_eq!(low.len(), n_rows);
                assert_eq!(high.len(), n_rows);
                assert!(
                    low.iter().zip_eq(high.iter()).all(|(&lo, &hi)| lo <= hi),
                    "target_low must be <= target_high for every row"
                );
            }
            (None, None) => {}
            _ => panic!("target_low and target_high must be supplied together"),
        }

        let avg_y = avg_y
            .unwrap_or_else(|| Self::compute_avg_y(y.as_slice().unwrap(), weights.as_ref().and_then(|w| w.as_slice())));

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights,
            variable_names,
            sequence_ids,
            target_low,
            target_high,
            avg_y,
        }
    }

    pub fn new(x: Array2<T>, y: Array1<T>) -> Self {
        Self::build_dataset(x, y, None, Vec::new(), None, None, None, None)
    }

    pub fn with_weights_and_names(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
    ) -> Self {
        Self::build_dataset(x, y, weights, variable_names, None, None, None, None)
    }

    pub fn with_weights_names_and_bounds(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        target_low: Array1<T>,
        target_high: Array1<T>,
    ) -> Self {
        Self::build_dataset(
            x,
            y,
            weights,
            variable_names,
            None,
            Some(target_low),
            Some(target_high),
            None,
        )
    }

    pub fn with_weights_names_and_sequence_ids(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        sequence_ids: Vec<usize>,
    ) -> Self {
        Self::build_dataset(x, y, weights, variable_names, Some(sequence_ids), None, None, None)
    }

    pub fn with_weights_names_sequence_ids_and_bounds(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        sequence_ids: Vec<usize>,
        target_low: Array1<T>,
        target_high: Array1<T>,
    ) -> Self {
        Self::build_dataset(
            x,
            y,
            weights,
            variable_names,
            Some(sequence_ids),
            Some(target_low),
            Some(target_high),
            None,
        )
    }

    pub fn y_slice(&self) -> &[T] {
        self.y.as_slice().expect("y is contiguous")
    }

    pub fn weights_slice(&self) -> Option<&[T]> {
        self.weights.as_ref().and_then(|w| w.as_slice())
    }

    pub fn sequence_ids_slice(&self) -> Option<&[usize]> {
        self.sequence_ids.as_deref()
    }

    pub fn target_bounds_slice(&self) -> Option<(&[T], &[T])> {
        match (&self.target_low, &self.target_high) {
            (Some(low), Some(high)) => Some((low.as_slice().unwrap(), high.as_slice().unwrap())),
            _ => None,
        }
    }

    pub fn delay_valid_at(&self, row: usize, offset: usize) -> bool {
        if row < offset {
            return false;
        }
        match self.sequence_ids.as_deref() {
            None => true,
            Some(ids) => ids[row] == ids[row - offset],
        }
    }

    pub fn has_valid_delay_rows(&self, offset: usize) -> bool {
        (0..self.n_rows).any(|row| self.delay_valid_at(row, offset))
    }

    pub fn compute_avg_y(y: &[T], weights: Option<&[T]>) -> T {
        if y.is_empty() {
            return T::zero();
        }
        match weights {
            None => {
                let n = T::from(y.len()).unwrap();
                y.iter().copied().fold(T::zero(), |a, b| a + b) / n
            }
            Some(w) => {
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                y.iter()
                    .copied()
                    .zip_eq(w.iter().copied())
                    .map(|(yi, wi)| yi * wi)
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w
            }
        }
    }

    pub fn make_batch_buffer(full: &Dataset<T>, batch_size: usize) -> Dataset<T> {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        let batch_size = batch_size.max(1);
        let x = Array2::<T>::zeros((full.n_features, batch_size));
        let y = Array1::<T>::zeros(batch_size);
        let weights = full.weights.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        let target_low = full.target_low.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        let target_high = full.target_high.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        Self::build_dataset(
            x,
            y,
            weights,
            full.variable_names.clone(),
            None,
            target_low,
            target_high,
            Some(full.avg_y),
        )
    }

    pub fn resample_from(&mut self, full: &Dataset<T>, rng: &mut Rng) {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        assert_eq!(self.n_features, full.n_features);
        assert_eq!(self.x.dim().0, self.n_features);
        assert_eq!(self.x.dim().1, self.n_rows);
        assert_eq!(self.y.len(), self.n_rows);
        if let Some(w) = &self.weights {
            assert_eq!(w.len(), self.n_rows);
            assert!(full.weights.is_some());
        } else {
            assert!(full.weights.is_none());
        }
        match (&self.target_low, &self.target_high) {
            (Some(low), Some(high)) => {
                assert_eq!(low.len(), self.n_rows);
                assert_eq!(high.len(), self.n_rows);
                assert!(full.target_low.is_some());
                assert!(full.target_high.is_some());
            }
            (None, None) => {
                assert!(full.target_low.is_none());
                assert!(full.target_high.is_none());
            }
            _ => panic!("target_low and target_high must be supplied together"),
        }

        for (dst_col, src_idx) in (0..self.n_rows).map(|i| (i, usize_range(rng, 0..full.n_rows))) {
            self.x.column_mut(dst_col).assign(&full.x.column(src_idx));
            self.y[dst_col] = full.y[src_idx];
            if let (Some(dst), Some(src)) = (self.weights.as_mut(), full.weights.as_ref()) {
                dst[dst_col] = src[src_idx];
            }
            if let (Some(dst), Some(src)) = (self.target_low.as_mut(), full.target_low.as_ref()) {
                dst[dst_col] = src[src_idx];
            }
            if let (Some(dst), Some(src)) = (self.target_high.as_mut(), full.target_high.as_ref()) {
                dst[dst_col] = src[src_idx];
            }
        }
    }
}

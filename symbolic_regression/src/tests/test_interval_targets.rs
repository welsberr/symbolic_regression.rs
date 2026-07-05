use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::dataset::TaggedDataset;
use crate::pop_member::{Evaluator, PopMember};
use crate::{Dataset, Options};

#[test]
fn interval_targets_drive_member_loss_when_enabled() {
    let x = Array2::from_shape_vec((1, 3), vec![0.5 as T, 4.0, 1.0]).unwrap();
    let y = Array1::from_vec(vec![0.0 as T, 0.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0 as T, 2.0, 1.0]);
    let target_low = Array1::from_vec(vec![0.0 as T, 1.0, 2.0]);
    let target_high = Array1::from_vec(vec![1.0 as T, 2.0, 3.0]);
    let dataset =
        Dataset::with_weights_names_and_bounds(x, y, Some(weights), vec!["x".into()], target_low, target_high);
    let tagged = TaggedDataset::new(&dataset, None);
    let expr = PostfixExpr::<T, TestOps, D>::new(
        vec![PNode::Var { feature: 0 }],
        Vec::new(),
        Metadata {
            variable_names: vec!["x".into()],
        },
    );
    let options = Options::<T, D> {
        use_interval_targets: true,
        progress: false,
        ..Default::default()
    };
    let mut member = PopMember::from_expr(expr, dataset.n_features, &options);
    let mut evaluator = Evaluator::new(dataset.n_rows);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    assert_eq!(member.loss, 9.0 / 4.0);
}

#[test]
#[should_panic(expected = "target_low must be <= target_high")]
fn interval_target_bounds_must_be_ordered() {
    let x = Array2::from_shape_vec((1, 1), vec![0.0 as T]).unwrap();
    let y = Array1::from_vec(vec![0.0 as T]);
    let target_low = Array1::from_vec(vec![2.0 as T]);
    let target_high = Array1::from_vec(vec![1.0 as T]);
    let _ = Dataset::with_weights_names_and_bounds(x, y, None, vec!["x".into()], target_low, target_high);
}

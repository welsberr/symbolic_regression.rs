use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::time::Instant;

use dynamic_expressions::{StringTreeOptions, delay_validity_mask};
use ndarray::{Array1, Array2};
use symbolic_regression::PopMember;
use symbolic_regression::prelude::*;

const D: usize = 3;

struct Args {
    train: String,
    test: Option<String>,
    target: String,
    features: Option<Vec<String>>,
    weight: Option<String>,
    target_low: Option<String>,
    target_high: Option<String>,
    sequence_id: Option<String>,
    niterations: usize,
    populations: usize,
    population_size: usize,
    cycles: usize,
    optimizer_iterations: usize,
    maxsize: usize,
    maxdepth: usize,
    max_delay: usize,
    delay_probability: f64,
    parsimony: f64,
    seed: u64,
    interval_targets: bool,
    unary_operators: Vec<String>,
    binary_operators: Vec<String>,
    selection: Selection,
}

#[derive(Clone)]
enum Selection {
    BestCost,
    BestLoss,
    ParetoIndex(usize),
    Complexity(usize),
}

struct Frame {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

struct Arrays {
    x: Array2<f32>,
    y: Array1<f32>,
    weights: Option<Array1<f32>>,
    target_low: Option<Array1<f32>>,
    target_high: Option<Array1<f32>>,
    sequence_ids: Option<Vec<usize>>,
}

fn main() {
    let args = parse_args();
    let train = read_csv(&args.train);
    let features = args
        .features
        .clone()
        .unwrap_or_else(|| infer_features(&train.headers, &args));
    let train_arrays = arrays(&train, &features, &args);
    let dataset = dataset_from_arrays(train_arrays, features.clone(), &args);
    let operator_names = operator_names(&args);
    let operators = BuiltinOpsF32::from_names(operator_names.iter().map(String::as_str)).unwrap();
    let options = Options::<f32, D> {
        seed: args.seed,
        niterations: args.niterations,
        populations: args.populations,
        population_size: args.population_size,
        ncycles_per_iteration: args.cycles,
        optimizer_iterations: args.optimizer_iterations,
        maxsize: args.maxsize,
        maxdepth: args.maxdepth,
        max_delay: args.max_delay,
        delay_probability: args.delay_probability,
        parsimony: args.parsimony,
        use_interval_targets: args.interval_targets,
        progress: false,
        deterministic: true,
        operators,
        ..Default::default()
    };

    let start = Instant::now();
    let result = equation_search::<f32, BuiltinOpsF32, D>(&dataset, &options);
    let elapsed = start.elapsed().as_secs_f64();
    let pareto = result.hall_of_fame.pareto_front();
    let (selected, selected_idx) = select_member(&pareto, &result.best, &args.selection);

    let best_expr = string_tree(
        &selected.expr,
        StringTreeOptions {
            variable_names: Some(&features),
            ..Default::default()
        },
    );

    println!("{{");
    println!("  \"engine\": \"sr.rs\",");
    println!("  \"status\": \"ok\",");
    println!("  \"wall_seconds\": {elapsed},");
    println!("  \"feature_columns\": {},", json_string_array(&features));
    println!("  \"unary_operators\": {},", json_string_array(&args.unary_operators));
    println!("  \"binary_operators\": {},", json_string_array(&args.binary_operators));
    println!("  \"selection\": {},", json_string(&args.selection.to_string()));
    println!("  \"train_loss\": {},", selected.loss);
    println!("  \"selected\": {{");
    if let Some(idx) = selected_idx {
        println!("    \"index\": {},", idx + 1);
    } else {
        println!("    \"index\": null,");
    }
    println!("    \"complexity\": {},", selected.complexity);
    println!("    \"loss\": {},", selected.loss);
    println!("    \"cost\": {},", selected.cost);
    println!("    \"expression\": {}", json_string(&best_expr));
    println!("  }},");

    if let Some(test_path) = &args.test {
        let test = read_csv(test_path);
        let test_arrays = arrays(&test, &features, &args);
        let (pred, complete) =
            eval_tree_array::<f32, BuiltinOpsF32, D>(&selected.expr, test_arrays.x.view(), &EvalOptions::default());
        let valid = delay_validity_mask(&selected.expr.nodes, pred.len(), test_arrays.sequence_ids.as_deref());
        let metrics = metrics(
            &pred,
            test_arrays.y.as_slice().unwrap(),
            test_arrays.target_low.as_ref().and_then(|a| a.as_slice()),
            test_arrays.target_high.as_ref().and_then(|a| a.as_slice()),
            &valid,
        );
        println!("  \"test\": {{");
        println!("    \"prediction_complete\": {complete},");
        println!("    \"valid_rows\": {},", metrics.valid_rows);
        println!("    \"mse\": {},", metrics.mse);
        println!("    \"r2\": {},", metrics.r2);
        if let Some(v) = metrics.interval_mse {
            println!("    \"interval_mse\": {v}");
        } else {
            println!("    \"interval_mse\": null");
        }
        println!("  }},");
    } else {
        println!("  \"test\": null,");
    }

    println!("  \"equations\": [");
    for (idx, member) in pareto.iter().enumerate() {
        let expr = string_tree(
            &member.expr,
            StringTreeOptions {
                variable_names: Some(&features),
                ..Default::default()
            },
        );
        let comma = if idx + 1 == pareto.len() { "" } else { "," };
        println!(
            "    {{\"index\": {}, \"is_selected\": {}, \"complexity\": {}, \"loss\": {}, \"cost\": {}, \"equation\": {}}}{comma}",
            idx + 1,
            Some(idx) == selected_idx,
            member.complexity,
            member.loss,
            member.cost,
            json_string(&expr)
        );
    }
    println!("  ]");
    println!("}}");
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut out = Args {
        train: String::new(),
        test: None,
        target: "target".into(),
        features: None,
        weight: None,
        target_low: None,
        target_high: None,
        sequence_id: None,
        niterations: 50,
        populations: 8,
        population_size: 80,
        cycles: 400,
        optimizer_iterations: 10,
        maxsize: 30,
        maxdepth: 8,
        max_delay: 0,
        delay_probability: 0.0,
        parsimony: 0.0,
        seed: 1009,
        interval_targets: false,
        unary_operators: vec!["cos".into(), "sin".into()],
        binary_operators: vec!["+".into(), "sub".into(), "*".into(), "/".into()],
        selection: Selection::BestCost,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => out.train = take_value(&mut args, "--train"),
            "--test" => out.test = Some(take_value(&mut args, "--test")),
            "--target" => out.target = take_value(&mut args, "--target"),
            "--features" => {
                out.features = Some(
                    take_value(&mut args, "--features")
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect(),
                )
            }
            "--weight" => out.weight = Some(take_value(&mut args, "--weight")),
            "--target-low" => out.target_low = Some(take_value(&mut args, "--target-low")),
            "--target-high" => out.target_high = Some(take_value(&mut args, "--target-high")),
            "--sequence-id" => out.sequence_id = Some(take_value(&mut args, "--sequence-id")),
            "--niterations" => out.niterations = parse_value(&mut args, "--niterations"),
            "--populations" => out.populations = parse_value(&mut args, "--populations"),
            "--population-size" => out.population_size = parse_value(&mut args, "--population-size"),
            "--cycles" => out.cycles = parse_value(&mut args, "--cycles"),
            "--optimizer-iterations" => out.optimizer_iterations = parse_value(&mut args, "--optimizer-iterations"),
            "--maxsize" => out.maxsize = parse_value(&mut args, "--maxsize"),
            "--maxdepth" => out.maxdepth = parse_value(&mut args, "--maxdepth"),
            "--max-delay" => out.max_delay = parse_value(&mut args, "--max-delay"),
            "--delay-probability" => out.delay_probability = parse_value(&mut args, "--delay-probability"),
            "--parsimony" => out.parsimony = parse_value(&mut args, "--parsimony"),
            "--seed" => out.seed = parse_value(&mut args, "--seed"),
            "--interval-targets" => out.interval_targets = true,
            "--unary-operators" => out.unary_operators = parse_operator_list(&mut args, "--unary-operators"),
            "--binary-operators" => out.binary_operators = parse_operator_list(&mut args, "--binary-operators"),
            "--selection" => out.selection = parse_selection(&take_value(&mut args, "--selection")),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    assert!(!out.train.is_empty(), "--train is required");
    if out.interval_targets {
        if out.target_low.is_none() {
            out.target_low = Some("target_low".into());
        }
        if out.target_high.is_none() {
            out.target_high = Some("target_high".into());
        }
    }
    out
}

fn print_help() {
    println!(
        "sr_rs --train TRAIN.csv [--test TEST.csv] [--target target] [--features a,b,c]\n\
         [--weight weight] [--target-low target_low --target-high target_high --interval-targets]\n\
         [--sequence-id seq] [--unary-operators cos,sin] [--binary-operators +,sub,*,/]\n\
         [--selection best-cost|best-loss|pareto-index=N|complexity=N]\n\
         [--max-delay N --delay-probability P] [search options]\n\n\
         Outputs JSON to stdout."
    );
}

fn take_value(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    args.next().unwrap_or_else(|| panic!("{name} requires a value"))
}

fn parse_value<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> T {
    take_value(args, name)
        .parse()
        .unwrap_or_else(|_| panic!("bad value for {name}"))
}

fn parse_operator_list(args: &mut impl Iterator<Item = String>, name: &str) -> Vec<String> {
    take_value(args, name)
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(normalize_operator_name)
        .collect()
}

fn normalize_operator_name(name: &str) -> String {
    match name {
        "-" => "sub".into(),
        other => other.into(),
    }
}

fn parse_selection(value: &str) -> Selection {
    match value {
        "best-cost" => Selection::BestCost,
        "best-loss" => Selection::BestLoss,
        _ if value.starts_with("pareto-index=") => {
            Selection::ParetoIndex(parse_selection_usize(value, "pareto-index="))
        }
        _ if value.starts_with("complexity=") => Selection::Complexity(parse_selection_usize(value, "complexity=")),
        _ => panic!("bad value for --selection: {value}"),
    }
}

fn parse_selection_usize(value: &str, prefix: &str) -> usize {
    value[prefix.len()..]
        .parse()
        .unwrap_or_else(|_| panic!("bad value for --selection: {value}"))
}

impl std::fmt::Display for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Selection::BestCost => write!(f, "best-cost"),
            Selection::BestLoss => write!(f, "best-loss"),
            Selection::ParetoIndex(idx) => write!(f, "pareto-index={idx}"),
            Selection::Complexity(complexity) => write!(f, "complexity={complexity}"),
        }
    }
}

fn operator_names(args: &Args) -> Vec<String> {
    args.unary_operators
        .iter()
        .chain(args.binary_operators.iter())
        .cloned()
        .collect()
}

fn select_member<'a>(
    pareto: &'a [PopMember<f32, BuiltinOpsF32, D>],
    fallback: &'a PopMember<f32, BuiltinOpsF32, D>,
    selection: &Selection,
) -> (&'a PopMember<f32, BuiltinOpsF32, D>, Option<usize>) {
    let selected_idx = match selection {
        Selection::BestCost => pareto
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost.total_cmp(&b.cost))
            .map(|(idx, _)| idx),
        Selection::BestLoss => pareto
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.loss.total_cmp(&b.loss))
            .map(|(idx, _)| idx),
        Selection::ParetoIndex(index) => {
            assert!(*index > 0, "pareto-index is 1-based and must be greater than zero");
            let idx = *index - 1;
            assert!(
                idx < pareto.len(),
                "pareto-index {index} is outside the equations table"
            );
            Some(idx)
        }
        Selection::Complexity(complexity) => pareto
            .iter()
            .position(|member| member.complexity == *complexity)
            .or_else(|| panic!("no equation with complexity {complexity}")),
    };

    selected_idx
        .map(|idx| (&pareto[idx], Some(idx)))
        .unwrap_or((fallback, None))
}

fn read_csv(path: &str) -> Frame {
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    let mut lines = text.lines();
    let headers: Vec<String> = lines
        .next()
        .unwrap_or_else(|| panic!("empty csv: {path}"))
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    Frame { headers, rows }
}

fn column_map(headers: &[String]) -> HashMap<&str, usize> {
    headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect()
}

fn infer_features(headers: &[String], args: &Args) -> Vec<String> {
    let mut excluded = BTreeSet::new();
    excluded.insert(args.target.as_str());
    if let Some(c) = &args.weight {
        excluded.insert(c.as_str());
    }
    if let Some(c) = &args.target_low {
        excluded.insert(c.as_str());
    }
    if let Some(c) = &args.target_high {
        excluded.insert(c.as_str());
    }
    if let Some(c) = &args.sequence_id {
        excluded.insert(c.as_str());
    }
    headers
        .iter()
        .filter(|name| !excluded.contains(name.as_str()))
        .cloned()
        .collect()
}

fn arrays(frame: &Frame, features: &[String], args: &Args) -> Arrays {
    let columns = column_map(&frame.headers);
    let n_rows = frame.rows.len();
    let mut x = Array2::<f32>::zeros((features.len(), n_rows));
    let mut y = Array1::<f32>::zeros(n_rows);
    let mut weights = args.weight.as_ref().map(|_| Array1::<f32>::zeros(n_rows));
    let mut target_low = args.target_low.as_ref().map(|_| Array1::<f32>::zeros(n_rows));
    let mut target_high = args.target_high.as_ref().map(|_| Array1::<f32>::zeros(n_rows));
    let mut sequence_ids = args.sequence_id.as_ref().map(|_| Vec::with_capacity(n_rows));
    let target_idx = *columns
        .get(args.target.as_str())
        .unwrap_or_else(|| panic!("missing target column {}", args.target));

    for (row_idx, row) in frame.rows.iter().enumerate() {
        for (feature_idx, feature) in features.iter().enumerate() {
            x[(feature_idx, row_idx)] = parse_cell(row, &columns, feature);
        }
        y[row_idx] = row[target_idx]
            .parse()
            .unwrap_or_else(|_| panic!("bad target value {}", row[target_idx]));
        if let (Some(col), Some(dst)) = (&args.weight, weights.as_mut()) {
            dst[row_idx] = parse_cell(row, &columns, col);
        }
        if let (Some(col), Some(dst)) = (&args.target_low, target_low.as_mut()) {
            dst[row_idx] = parse_cell(row, &columns, col);
        }
        if let (Some(col), Some(dst)) = (&args.target_high, target_high.as_mut()) {
            dst[row_idx] = parse_cell(row, &columns, col);
        }
        if let (Some(col), Some(dst)) = (&args.sequence_id, sequence_ids.as_mut()) {
            let idx = *columns
                .get(col.as_str())
                .unwrap_or_else(|| panic!("missing sequence column {col}"));
            dst.push(
                row[idx]
                    .parse()
                    .unwrap_or_else(|_| panic!("bad sequence id {}", row[idx])),
            );
        }
    }

    Arrays {
        x,
        y,
        weights,
        target_low,
        target_high,
        sequence_ids,
    }
}

fn parse_cell(row: &[String], columns: &HashMap<&str, usize>, name: &str) -> f32 {
    let idx = *columns.get(name).unwrap_or_else(|| panic!("missing column {name}"));
    row[idx]
        .parse()
        .unwrap_or_else(|_| panic!("bad float in column {name}: {}", row[idx]))
}

fn dataset_from_arrays(arrays: Arrays, features: Vec<String>, args: &Args) -> Dataset<f32> {
    match (arrays.sequence_ids, arrays.target_low, arrays.target_high) {
        (Some(sequence_ids), Some(low), Some(high)) => Dataset::with_weights_names_sequence_ids_and_bounds(
            arrays.x,
            arrays.y,
            arrays.weights,
            features,
            sequence_ids,
            low,
            high,
        ),
        (Some(sequence_ids), None, None) => {
            Dataset::with_weights_names_and_sequence_ids(arrays.x, arrays.y, arrays.weights, features, sequence_ids)
        }
        (None, Some(low), Some(high)) => {
            Dataset::with_weights_names_and_bounds(arrays.x, arrays.y, arrays.weights, features, low, high)
        }
        (None, None, None) => Dataset::with_weights_and_names(arrays.x, arrays.y, arrays.weights, features),
        _ if args.interval_targets => panic!("target_low and target_high must both be available"),
        _ => panic!("invalid target-bound state"),
    }
}

struct Metrics {
    valid_rows: usize,
    mse: f64,
    r2: f64,
    interval_mse: Option<f64>,
}

fn metrics(pred: &[f32], y: &[f32], low: Option<&[f32]>, high: Option<&[f32]>, valid: &[bool]) -> Metrics {
    assert_eq!(pred.len(), y.len());
    assert_eq!(pred.len(), valid.len());
    let valid_rows = valid.iter().filter(|&&v| v).count();
    if valid_rows == 0 {
        return Metrics {
            valid_rows: 0,
            mse: f64::NAN,
            r2: f64::NAN,
            interval_mse: None,
        };
    }

    let mean = y
        .iter()
        .zip(valid)
        .filter_map(|(&target, &ok)| ok.then_some(target as f64))
        .sum::<f64>()
        / valid_rows as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut interval = 0.0;
    for row in 0..pred.len() {
        if !valid[row] {
            continue;
        }
        let err = pred[row] as f64 - y[row] as f64;
        ss_res += err * err;
        let terr = y[row] as f64 - mean;
        ss_tot += terr * terr;
        if let (Some(low), Some(high)) = (low, high) {
            let ierr = if pred[row] < low[row] {
                pred[row] as f64 - low[row] as f64
            } else if pred[row] > high[row] {
                pred[row] as f64 - high[row] as f64
            } else {
                0.0
            };
            interval += ierr * ierr;
        }
    }

    Metrics {
        valid_rows,
        mse: ss_res / valid_rows as f64,
        r2: 1.0 - ss_res / ss_tot,
        interval_mse: low.zip(high).map(|_| interval / valid_rows as f64),
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_string_array(values: &[String]) -> String {
    let parts = values.iter().map(|v| json_string(v)).collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

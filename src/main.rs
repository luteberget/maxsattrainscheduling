use ddd::{
    maxsat,
    problem::{DelayCostType, DelayMeasurementType},
};
use std::process::exit;

const TIMEOUT: f64 = 120.0;
const DELAY_MEASUREMENT_TYPE: DelayMeasurementType = DelayMeasurementType::FinalStationArrival;
const DELAY_COST_TYPE: DelayCostType = DelayCostType::InfiniteSteps180;

fn main() {
    pretty_env_logger::env_logger::Builder::from_env(
        pretty_env_logger::env_logger::Env::default().default_filter_or("trace"),
    )
    .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 1 {
        log::error!("Usage: trainscheduling <filename>");
        exit(1);
    }

    let filename = &args[0];

    log::debug!("Reading {}", filename);
    #[allow(unused)]
    let (named_problem, _) =
        ddd::parser::read_txt_file(&filename, DELAY_MEASUREMENT_TYPE, false, None, |_| {});

    let solution = ddd::scheduling::solve(
        maxsat::IPAMIRSolver::new(),
        &named_problem.problem,
        TIMEOUT,
        DELAY_COST_TYPE,
    );

    let cost = named_problem
        .problem
        .verify_solution(&solution.unwrap().0, DELAY_COST_TYPE)
        .unwrap();

    println!("success: cost={}", cost);
}

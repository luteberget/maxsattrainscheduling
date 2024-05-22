use std::process::exit;

use ipamir_trainscheduling::problem::{DelayCostType, DelayMeasurementType};
use log::info;

const TIMEOUT: f64 = f64::INFINITY;
const DELAY_MEASUREMENT_TYPE: DelayMeasurementType = DelayMeasurementType::FinalStationArrival;
const DELAY_COST_TYPE: DelayCostType = DelayCostType::InfiniteSteps180;

fn main() {
    pretty_env_logger::env_logger::Builder::from_env(
        pretty_env_logger::env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        log::error!("Usage: {} <filename>", args[0]);
        exit(1);
    }

    let filename = &args[1];

    log::debug!("Reading {}", filename);
    #[allow(unused)]
    let (named_problem, _) =
        ipamir_trainscheduling::parser::read_txt_file(&filename, DELAY_MEASUREMENT_TYPE, false, None, |_| {});

    let solution = ipamir_trainscheduling::scheduling::solve(
        ipamir_trainscheduling::ipamir::IPAMIRSolver::new(),
        &named_problem.problem,
        TIMEOUT,
        DELAY_COST_TYPE,
    );

    let cost = named_problem
        .problem
        .verify_solution(&solution.unwrap().0, DELAY_COST_TYPE)
        .unwrap();

    info!("success: cost={}", cost);
}

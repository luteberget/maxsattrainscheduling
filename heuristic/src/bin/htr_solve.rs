use heuristic::{
    problem::convert_ddd_problem,
    solvers::train_queue::QueueTrainSolver,
    solvers::{solver_brb::BnBConflictSolver, solver_mcts},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "htr_solve")]
struct Opt {
    #[structopt(short, long)]
    print_table: bool,

    #[structopt(name = "MODE")]
    mode: String,

    /// Files to process
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,

    /// Time limit (milliseconds)
    #[structopt(short, long, name = "TIME")]
    time_limit: Option<u128>,
}

pub fn main() {
    pretty_env_logger::init();

    let opt = Opt::from_args();
    println!("{:#?}", opt);

    enum SolverMode {
        First,
        Exact,
        HeurHeur,
        HeurHeur2,
        Random,
        Uct,
    }

    let mode = match opt.mode.as_str() {
        "exact" => SolverMode::Exact,
        "first" => SolverMode::First,
        "heurheur" => SolverMode::HeurHeur,
        "heurheur2" => SolverMode::HeurHeur2,
        "random" => SolverMode::Random,
        "uct" => SolverMode::Uct,
        _ => panic!("unknown solver mode"),
    };

    if opt.files.is_empty() {
        panic!("No input files specified");
    }

    let mut results = Vec::new();
    struct Result {
        instance: String,
        cost: i32,
        solve_time: Duration,
    }

    for file in opt.files {
        hprof::start_frame();
        println!("Solving {}", file.to_string_lossy());

        let (ddd_problem, _) = ddd_problem::parser::read_txt_file(
            &file.to_string_lossy(),
            ddd_problem::problem::DelayMeasurementType::FinalStationArrival,
            false,
            None,
            |_| {},
        );

        let htr_problem = convert_ddd_problem(&ddd_problem);

        let start_time = Instant::now();
        let (cost, solution) = match mode {
            SolverMode::Exact => {
                let _p = hprof::enter("solve exact");
                let mut solver = BnBConflictSolver::<QueueTrainSolver>::new(htr_problem);
                let mut best = (i32::MAX, None);
                let mut n_solutions = 0;
                while let Some((cost, sol)) = solver.solve_next_stopcb(|| {
                    start_time.elapsed().as_millis() >= opt.time_limit.unwrap_or(u128::MAX)
                }) {
                    if cost < best.0 {
                        best = (cost, Some(sol));
                        println!("{}", cost);
                        println!(
                            "solutions:{} nodes_created:{} nodes_explored:{}",
                            n_solutions,
                            solver.conflict_space.n_nodes_generated,
                            solver.conflict_space.n_nodes_explored
                        );
                    }
                    n_solutions += 1;
                }

                println!(
                    "solutions:{} nodes_created:{} nodes_explored:{}",
                    n_solutions,
                    solver.conflict_space.n_nodes_generated,
                    solver.conflict_space.n_nodes_explored
                );
                (best.0, best.1.unwrap())
            }
            SolverMode::First => {
                let _p = hprof::enter("solve first");
                let mut solver = BnBConflictSolver::<QueueTrainSolver>::new(htr_problem);
                solver.solve_next().unwrap()
            }
            SolverMode::HeurHeur => {
                let _p = hprof::enter("solve heurheur");
                let mut solver = heuristic::solvers::solver_heurheur::HeurHeur::new(htr_problem);
                solver.solve().unwrap()
            }
            SolverMode::Uct => {
                let _p = hprof::enter("solve uct");
                solver_mcts::solve(htr_problem, || {
                    start_time.elapsed().as_millis() >= opt.time_limit.unwrap_or(u128::MAX)
                })
                .unwrap()
            }
            SolverMode::HeurHeur2 => {
                let _p = hprof::enter("solve heurheur2");
                let mut solver = heuristic::solvers::solver_heurheur2::HeurHeur2::new(htr_problem);
                solver.solve().unwrap()
            }
            SolverMode::Random => {
                let _p = hprof::enter("solve random");
                let mut solver =
                    heuristic::solvers::solver_random::RandomHeuristic::new(htr_problem);
                solver.solve().unwrap()
            }
        };
        let solve_time = start_time.elapsed();
        println!(
            "output size: {}, COST: {}",
            solution.iter().map(|s| s.len()).sum::<usize>(),
            cost,
        );
        let cost = ddd_problem
            .problem
            .verify_solution(&solution, ddd_problem::problem::DelayCostType::Continuous)
            .unwrap();

        hprof::profiler().print_timing();

        results.push(Result {
            cost,
            instance: file.to_string_lossy().to_string(),
            solve_time,
        });
    }

    if opt.print_table {
        for result in results {
            println!(
                "{}  {}  {:.0}",
                result.instance,
                result.cost,
                result.solve_time.as_secs_f64() * 1000.0,
            );
        }
    }
}

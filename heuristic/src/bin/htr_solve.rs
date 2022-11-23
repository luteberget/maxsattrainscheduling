use heuristic::{problem::convert_ddd_problem, solver::ConflictSolver};
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
}

pub fn main() {
    let opt = Opt::from_args();
    println!("{:#?}", opt);

    enum SolverMode {
        Exact,
    }

    let mode = match opt.mode.as_str() {
        "exact" => SolverMode::Exact,
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
        let solution = match mode {
            SolverMode::Exact => {
                let _p = hprof::enter("solve exact");
                let solver = ConflictSolver::new(htr_problem);
                solver.solve_any()
            }
        };
        
        let solve_time = start_time.elapsed();
        println!(
            "output size: {}",
            solution.iter().map(|s| s.len()).sum::<usize>()
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
                result.solve_time.as_secs_f64()*1000.0,
            );
        }
    }
}

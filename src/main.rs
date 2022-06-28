use std::{cell::RefCell, collections::HashSet, fmt::Write};

use ddd::{
    parser,
    problem::{self, DelayCostThresholds, DelayCostType, NamedProblem, Visit},
    solvers::{bigm, maxsatddd, SolverError},
};

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "trainscheduler",
    about = "Optimal train scheduling experiments."
)]
struct Opt {
    /// Activate debug mode
    #[structopt(short, long)]
    debug: bool,

    #[structopt(short, long)]
    solvers: Vec<String>,

    #[structopt(long)]
    xml_instances: bool,

    #[structopt(long)]
    txt_instances: bool,

    #[structopt(long)]
    instance_name_filter: Option<String>,

    #[structopt(long)]
    verify_instances: bool,

    #[structopt(long)]
    objective: Option<String>,

    #[structopt(long)]
    other_objective: Option<String>,
}

pub fn xml_instances(mut x: impl FnMut(String, NamedProblem)) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for instance_id in a_instances
        .into_iter()
        .chain(b_instances)
        .chain(c_instances)
    {
        let filename = format!("instances/Instance{}.xml", instance_id);
        println!("Reading {}", filename);
        #[allow(unused)]
        let problem = parser::read_xml_file(
            &filename,
            problem::DelayMeasurementType::FinalStationArrival,
        );
        x(format!("xml {}", instance_id), problem);
    }
}

pub fn txt_instances(mut x: impl FnMut(String, NamedProblem)) {
    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    // #[allow(unused)]
    // let c_instances = [21, 22, 23, 24];

    for (dir, shortname) in [
        ("instances_original", "orig"),
        ("instances_addtracktime", "track"),
        ("instances_addstationtime", "station"),
    ] {
        let instances = ["A", "B"]
            .iter()
            .flat_map(move |n| (1..=12).map(move |i| (n, i)));

        // let instances = instances.skip(16).take(1);

        for (infrastructure, number) in instances {
            let filename = format!("{}/Instance{}{}.txt", dir, infrastructure, number);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, _) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                false,
                None,
                |_| {},
            );
            x(
                format!("{}{}{}", shortname, infrastructure, number),
                problem,
            );
        }
    }
}

pub fn verify_instances(mut x: impl FnMut(String, NamedProblem, Vec<Vec<i32>>) -> Vec<Vec<i32>>) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];
    let instances = [20];

    for solvertype in ["BigMComplete", "BigMLazyCon"] {
        for instance_id in a_instances
            .iter()
            .chain(b_instances.iter())
            .chain(c_instances.iter())
        {
            let filename = format!("InstanceResults/{}Sol{}.txt", solvertype, instance_id);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, solution) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                true,
                None,
                |_| {},
            );
            let new_solution = x(
                format!("{} {}", solvertype, instance_id),
                problem,
                solution.unwrap(),
            );

            // let mut f = std::fs::File::create(&format!("{}.bl.txt", filename)).unwrap();
            // use std::io::Write;
            // parser::read_txt_file(
            //     &filename,
            //     problem::DelayMeasurementType::FinalStationArrival,
            //     true,
            //     Some(new_solution),
            //     |l| {
            //         writeln!(f, "{}", l).unwrap();
            //     },
            // );
        }
    }
}

#[derive(Debug)]
enum SolverType {
    BigMEager,
    BigMLazy,
    MaxSatDdd,
    MaxSatDddCadical,
    MaxSatIdl,
    MipDdd,
    MipHull,
}

fn main() {
    pretty_env_logger::env_logger::Builder::from_env(
        pretty_env_logger::env_logger::Env::default().default_filter_or("trace"),
    )
    .init();

    let opt = Opt::from_args();
    println!("{:?}", opt);
    let solvers = opt
        .solvers
        .iter()
        .map(|x| match x.as_str() {
            "bigm_eager" => SolverType::BigMEager,
            "bigm_lazy" => SolverType::BigMLazy,
            "maxsat_ddd" => SolverType::MaxSatDdd,
            "maxsat_ddd_cdc" => SolverType::MaxSatDddCadical,
            "maxsat_idl" => SolverType::MaxSatIdl,
            "mip_ddd" => SolverType::MipDdd,
            "mip_hull" => SolverType::MipHull,
            _ => panic!("unknown solver type"),
        })
        .collect::<Vec<_>>();
    println!("Using solvers {:?}", solvers);

    let delay_cost_type = opt
        .objective
        .map(|obj| match obj.as_str() {
            "finsteps123" => DelayCostType::FiniteSteps123,
            "finsteps12345" => DelayCostType::FiniteSteps12345,
            "finsteps139" => DelayCostType::FiniteSteps139,
            "infsteps60" => DelayCostType::InfiniteSteps60,
            "infsteps180" => DelayCostType::InfiniteSteps180,
            "infsteps360" => DelayCostType::InfiniteSteps360,
            "cont" => DelayCostType::Continuous,
            _ => panic!("Unknown objective type."),
        })
        .unwrap_or(DelayCostType::FiniteSteps123);
    println!("Using delay cost type {:?}", delay_cost_type);

    let other_delay_cost_type = opt.other_objective.map(|obj| match obj.as_str() {
        "finsteps123" => DelayCostType::FiniteSteps123,
        "finsteps12345" => DelayCostType::FiniteSteps12345,
        "finsteps139" => DelayCostType::FiniteSteps139,
        "infsteps60" => DelayCostType::InfiniteSteps60,
        "infsteps180" => DelayCostType::InfiniteSteps180,
        "infsteps360" => DelayCostType::InfiniteSteps360,
        "cont" => DelayCostType::Continuous,
        _ => panic!("Unknown objective type."),
    });

    let perf_out = RefCell::new(String::new());

    println!("Starting gurobi environment...");
    let mut env = grb::Env::new("").unwrap();
    env.set(grb::param::OutputFlag, 0).unwrap();
    println!("...ok.");

    let solve_it = |name: String, p: NamedProblem| -> Result<Vec<Vec<i32>>, SolverError> {
        if solvers.is_empty() {
            panic!("no solver specified");
        }
        let problemstats = print_problem_stats(&p.problem);

        let mut solution = Result::Err(SolverError::NoSolution);
        for solver in solvers.iter() {
            hprof::start_frame();
            println!("Starting solver {:?}", solver);
            solution = match solver {
                SolverType::BigMEager => bigm::solve_bigm(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    false,
                    30.0,
                    &p.train_names,
                    &p.resource_names,
                ),
                SolverType::MipHull => bigm::solve_hull(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    true,
                    30.0,
                    &p.train_names,
                    &p.resource_names,
                ),
                SolverType::BigMLazy => bigm::solve_bigm(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    true,
                    30.0,
                    &p.train_names,
                    &p.resource_names,
                ),
                SolverType::MaxSatDdd => maxsatddd::solve(
                    &env,
                    satcoder::solvers::minisat::Solver::new(),
                    &p.problem,
                    30.0,
                    delay_cost_type,
                )
                .map(|(v, _)| v),
                SolverType::MaxSatDddCadical => maxsatddd::solve(
                    &env,
                    satcoder::solvers::cadical::Solver::new(),
                    &p.problem,
                    30.0,
                    delay_cost_type,
                )
                .map(|(v, _)| v),
                SolverType::MaxSatIdl => {
                    if let DelayCostType::FiniteSteps123 = delay_cost_type {
                        ddd::solvers::idl::solve(&p.problem)
                    } else {
                        panic!("Unsupported delay cost type for IDL solver.");
                    }
                }
                SolverType::MipDdd => {
                    if let DelayCostType::FiniteSteps123 = delay_cost_type {
                        ddd::solvers::mipdddpack::solve(&env, &p.problem, delay_cost_type)
                    } else {
                        panic!("Unsupported delay cost type for MipDDD solver.")
                    }
                }
            };
            hprof::end_frame();
            let solver_name = format!("{:?}", solver);

            if let Ok(solution) = solution.as_ref() {
                let cost = p
                    .problem
                    .verify_solution(solution, delay_cost_type)
                    .unwrap();

                let other_cost =
                    other_delay_cost_type.map(|c| p.problem.verify_solution(solution, c).unwrap());

                hprof::profiler().print_timing();
                let _root = hprof::profiler().root();
                let sol_time = hprof::profiler().root().total_time.get() as f64 / 1_000_000f64;
                writeln!(
                    perf_out.borrow_mut(),
                    "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                    name,
                    delay_cost_type,
                    solver_name,
                    cost,
                    sol_time,
                )
                .unwrap();

                if let Some(other_cost) = other_cost {
                    writeln!(
                        perf_out.borrow_mut(),
                        "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                        name,
                        other_delay_cost_type,
                        solver_name,
                        other_cost,
                        sol_time,
                    )
                    .unwrap();
                }
            } else {
                writeln!(
                    perf_out.borrow_mut(),
                    "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                    name,
                    delay_cost_type,
                    solver_name,
                    9999.0,
                    9999.0,
                )
                .unwrap();
                if other_delay_cost_type.is_some() {
                    writeln!(
                        perf_out.borrow_mut(),
                        "{:>10} {:<25?} {:<12} {:>5} {:>10.0}",
                        name,
                        delay_cost_type,
                        solver_name,
                        9999.0,
                        9999.0,
                    )
                    .unwrap();
                }
            }
        }
        solution
    };

    if opt.xml_instances {
        xml_instances(|name, p| {
            solve_it(name, p);
        });
    }
    if opt.txt_instances {
        txt_instances(|name, p| {
            solve_it(name, p);
        });
    }
    if opt.verify_instances {
        verify_instances(|name, p, solution| {
            let cost = p
                .problem
                .verify_solution(&solution, delay_cost_type)
                .unwrap();
            writeln!(perf_out.borrow_mut(), "{:>10} {:>5}", name, cost,).unwrap();
            solve_it(name, p).unwrap()
        })
    }
    println!("{}", perf_out.into_inner());
}

struct ProblemStats {
    trains: usize,
    conflicts: usize,
    avg_tracks: f32,
    conflicting_visit_pairs: usize,
}

fn print_problem_stats(problem: &problem::Problem) -> ProblemStats {
    let avg_tracks = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .filter(|v| problem.conflicts.contains(&(v.resource_id, v.resource_id)))
                .count()
        })
        .sum::<usize>() as f32
        / problem.trains.len() as f32;
    let mut conflicting_visit_pairs = 0;
    for t1 in 0..problem.trains.len() {
        for t2 in (t1 + 1)..problem.trains.len() {
            for Visit {
                resource_id: r1, ..
            } in problem.trains[t1].visits.iter()
            {
                for Visit {
                    resource_id: r2, ..
                } in problem.trains[t2].visits.iter()
                {
                    if problem.conflicts.contains(&(*r1, *r2)) {
                        conflicting_visit_pairs += 1;
                    }
                }
            }
        }
    }

    let delays = 0;
    let avgdelay = 0;

    let trains = problem.trains.len();
    let conflicts = problem.conflicts.len();
    println!(
        "trains {} tracks {} avgtracks {:.2} trackpairs {} delays {} avgdelay{}",
        trains, conflicts, avg_tracks, conflicting_visit_pairs, delays, avgdelay,
    );
    ProblemStats {
        trains,
        conflicts,
        avg_tracks,
        conflicting_visit_pairs,
    }
}

#[cfg(test)]
mod tests {
    use ddd::problem::{DelayCostType, NamedProblem};

    #[test]
    pub fn testproblem_maxsatddd() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;

        let problem = crate::problem::problem1_with_stations();
        let result = ddd::solvers::maxsatddd::solve(
            &env,
            satcoder::solvers::minisat::Solver::new(),
            &problem,
            30.0,
            DelayCostType::FiniteSteps123,
        )
        .unwrap()
        .0;
        let score = problem.verify_solution(&result, delay_cost_type);
        assert!(score.is_some());
    }

    #[test]
    pub fn testproblem_mipdddpack() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let problem = crate::problem::problem1_with_stations();
        let result = ddd::solvers::mipdddpack::solve(&env, &problem, delay_cost_type).unwrap();
        let score = problem.verify_solution(&result, delay_cost_type);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore_trivial() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let problem = crate::problem::problem1_with_stations();
        let delay_cost_type = DelayCostType::FiniteSteps123;

        let result = ddd::solvers::maxsatddd::solve(
            &env,
            satcoder::solvers::minisat::Solver::new(),
            &problem,
            30.0,
            delay_cost_type,
        )
        .unwrap()
        .0;
        let first_score = problem.verify_solution(&result, delay_cost_type);

        for _ in 0..100 {
            let result = ddd::solvers::maxsatddd::solve(
                &env,
                satcoder::solvers::minisat::Solver::new(),
                &problem,
                30.0,
                DelayCostType::FiniteSteps123,
            )
            .unwrap()
            .0;
            let score = problem.verify_solution(&result, delay_cost_type);
            assert!(score == first_score);
        }
        println!("ALL COSTS WERE {:?}", first_score);
    }

    #[test]
    pub fn samescore_all_instances() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let delay_cost_type = DelayCostType::FiniteSteps123;
        for delaytype in [
            ddd::problem::DelayMeasurementType::AllStationArrivals,
            ddd::problem::DelayMeasurementType::FinalStationArrival,
        ] {
            for instance_number in
                // [                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,            ]
                [3, 4, 5]
            {
                println!("{}", instance_number);
                let NamedProblem { problem, .. } = crate::parser::read_xml_file(
                    &format!("instances/Instance{}.xml", instance_number,),
                    delaytype,
                );

                let result = ddd::solvers::maxsatddd::solve(
                    &env,
                    satcoder::solvers::minisat::Solver::new(),
                    &problem,
                    30.0,
                    delay_cost_type,
                )
                .unwrap()
                .0;
                let first_score = problem.verify_solution(&result, delay_cost_type);

                for iteration in 0..100 {
                    println!("iteration {} {}", instance_number, iteration);
                    let result = ddd::solvers::maxsatddd::solve(
                        &env,
                        satcoder::solvers::minisat::Solver::new(),
                        &problem,
                        30.0,
                        DelayCostType::FiniteSteps123,
                    )
                    .unwrap()
                    .0;
                    let score = problem.verify_solution(&result, delay_cost_type);
                    assert!(score == first_score);
                }

                println!("ALL COSTS WERE {:?}", first_score);
            }
        }
    }
}

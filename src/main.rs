use std::{cell::RefCell, collections::HashSet, fmt::Write};

use ddd::{
    parser,
    problem::{self, DelayCostThresholds, NamedProblem, Visit, DelayCostType},
    solvers::{bigm, maxsatddd},
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
}

pub fn xml_instances(mut x: impl FnMut(String, NamedProblem)) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for instance_id in a_instances.into_iter()
    // .chain(b_instances)
    // .chain(c_instances)
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
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for (dir, shortname) in [("txtinstances", "txt1"), ("txtinstances2", "txt2")] {
        for instance_id in a_instances
            .into_iter()
            .chain(b_instances)
            .chain(c_instances)
        {
            let filename = format!("{}/Instance{}.txt", dir, instance_id);
            println!("Reading {}", filename);
            #[allow(unused)]
            let (problem, _) = parser::read_txt_file(
                &filename,
                problem::DelayMeasurementType::FinalStationArrival,
                false,
                None,
                |_| {},
            );
            x(format!("{} {}", shortname, instance_id), problem);
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
    println!("Using solvers {:?}", opt.solvers);
    let solvers = opt
        .solvers
        .iter()
        .map(|x| match x.as_str() {
            "bigm_eager" => SolverType::BigMEager,
            "bigm_lazy" => SolverType::BigMLazy,
            "maxsat_ddd" => SolverType::MaxSatDdd,
            "maxsat_idl" => SolverType::MaxSatIdl,
            "mip_ddd" => SolverType::MipDdd,
            "mip_hull" => SolverType::MipHull,
            _ => panic!("unknown solver type"),
        })
        .collect::<Vec<_>>();

    let delay_cost_type = opt
        .objective
        .map(|obj| match obj.as_str() {
            "step123" => DelayCostType::Step123,
            "cont" => DelayCostType::Continuous,
            _ => panic!("Unknown objective type."),
        })
        .unwrap_or(DelayCostType::Step123);

    let perf_out = RefCell::new(String::new());

    let mut env = grb::Env::new("").unwrap();
    env.set(grb::param::OutputFlag, 0).unwrap();

    let solve_it = |name: String, p: NamedProblem| -> Vec<Vec<i32>> {
        if solvers.is_empty() {
            panic!("no solver specified");
        }
        let problemstats = print_problem_stats(&p.problem);

        let mut solution = Vec::new();
        for solver in solvers.iter() {
            hprof::start_frame();
            println!("Starting solver {:?}", solver);
            solution = match solver {
                SolverType::BigMEager => bigm::solve_bigm(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    false,
                    &p.train_names,
                    &p.resource_names,
                )
                .unwrap(),
                SolverType::MipHull => bigm::solve_hull(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    true,
                    &p.train_names,
                    &p.resource_names,
                )
                .unwrap(),
                SolverType::BigMLazy => bigm::solve_bigm(
                    &env,
                    &p.problem,
                    delay_cost_type,
                    true,
                    &p.train_names,
                    &p.resource_names,
                )
                .unwrap(),
                SolverType::MaxSatDdd => {
                    if let DelayCostType::Step123 = delay_cost_type {
                        maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &p.problem)
                            .unwrap()
                            .0
                    } else {
                        panic!("Unsupported delay cost type for MaxSATDDD solver.");
                    }
                }
                SolverType::MaxSatIdl => {
                    if let DelayCostType::Step123 = delay_cost_type {
                        ddd::solvers::idl::solve(&p.problem).unwrap()
                    } else {
                        panic!("Unsupported delay cost type for IDL solver.");
                    }
                }
                SolverType::MipDdd => {
                    if let DelayCostType::Step123 = delay_cost_type {
                        ddd::solvers::mipdddpack::solve(&env, &p.problem).unwrap()
                    } else {
                        panic!("Unsupported delay cost type for MipDDD solver.")
                    }
                }
            };
            hprof::end_frame();

            let cost = p.problem.verify_solution(&solution).unwrap();
            hprof::profiler().print_timing();
            let _root = hprof::profiler().root();
            let sol_time = hprof::profiler().root().total_time.get() as f64 / 1_000_000f64;
            let solver_name = format!("{:?}", solver);
            writeln!(
                perf_out.borrow_mut(),
                "{:>10} {:<12} {:>5} {:>10.0}",
                name,
                solver_name,
                cost,
                sol_time,
            )
            .unwrap();
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
            let cost = p.problem.verify_solution(&solution).unwrap();
            writeln!(perf_out.borrow_mut(), "{:>10} {:>5}", name, cost,).unwrap();
            solve_it(name, p)
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
    use ddd::problem::NamedProblem;

    #[test]
    pub fn testproblem_maxsatddd() {
        let problem = crate::problem::problem1_with_stations();
        let result =
            ddd::solvers::maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                .unwrap()
                .0;
        let score = problem.verify_solution(&result);
        assert!(score.is_some());
    }

    #[test]
    pub fn testproblem_mipdddpack() {
        let mut env = grb::Env::new("").unwrap();
        env.set(grb::param::OutputFlag, 0).unwrap();

        let problem = crate::problem::problem1_with_stations();
        let result = ddd::solvers::mipdddpack::solve(&env, &problem).unwrap();
        let score = problem.verify_solution(&result);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore_trivial() {
        let problem = crate::problem::problem1_with_stations();

        let result =
            ddd::solvers::maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                .unwrap()
                .0;
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result =
                ddd::solvers::maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                    .unwrap()
                    .0;
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
        println!("ALL COSTS WERE {:?}", first_score);
    }

    #[test]
    pub fn samescore_all_instances() {
        for delaytype in [
            ddd::problem::DelayMeasurementType::AllStationArrivals,
            ddd::problem::DelayMeasurementType::FinalStationArrival,
        ] {
            for instance_number in [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            ] {
                println!("{}", instance_number);
                let NamedProblem { problem, .. } = crate::parser::read_xml_file(
                    &format!("instances/Instance{}.xml", instance_number,),
                    delaytype,
                );

                let result = ddd::solvers::maxsatddd::solve(
                    satcoder::solvers::minisat::Solver::new(),
                    &problem,
                )
                .unwrap()
                .0;
                let first_score = problem.verify_solution(&result);

                for iteration in 0..100 {
                    println!("iteration {} {}", instance_number, iteration);
                    let result = ddd::solvers::maxsatddd::solve(
                        satcoder::solvers::minisat::Solver::new(),
                        &problem,
                    )
                    .unwrap()
                    .0;
                    let score = problem.verify_solution(&result);
                    assert!(score == first_score);
                }

                println!("ALL COSTS WERE {:?}", first_score);
            }
        }
    }
}

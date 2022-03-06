use std::{collections::HashSet, fmt::Write};

use ddd::{
    parser,
    problem::{self, DelayCostThresholds, NamedProblem, Visit},
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
}

pub fn xml_instances(mut x: impl FnMut(String, NamedProblem)) {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
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
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    for instance_id in a_instances
        .into_iter()
        .chain(b_instances)
        .chain(c_instances)
    {
        let filename = format!("txtinstances/Instance{}.txt", instance_id);
        println!("Reading {}", filename);
        #[allow(unused)]
        let problem = parser::read_txt_file(
            &filename,
            problem::DelayMeasurementType::FinalStationArrival,
        );
        x(format!("txt {}", instance_id), problem);
    }
}


#[derive(Debug)]
enum SolverType {
    BigMEager,
    BigMLazy,
    MaxSatDdd,
    MaxSatIdl,
}

fn main() {
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
            _ => panic!("unknown solver type"),
        })
        .collect::<Vec<_>>();

    if solvers.is_empty() {
        panic!("no solver specified");
    }

    let mut perf_out = String::new();

    let mut solve_it = |name: String, p: NamedProblem| {
        let problemstats = print_problem_stats(&p.problem);

        for solver in solvers.iter() {
            hprof::start_frame();
            println!("Starting solver {:?}", solver);
            let solution = match solver {
                SolverType::BigMEager => bigm::solve(&p.problem, false).unwrap(),
                SolverType::BigMLazy => bigm::solve(&p.problem, true).unwrap(),
                SolverType::MaxSatDdd => {
                    maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &p.problem)
                        .unwrap()
                        .0
                }
                SolverType::MaxSatIdl => ddd::solvers::idl::solve(&p.problem).unwrap(),
            };
            hprof::end_frame();

            let cost = p.problem.verify_solution(&solution).unwrap();
            hprof::profiler().print_timing();
            let _root = hprof::profiler().root();
            let sol_time = hprof::profiler().root().total_time.get() as f64 / 1_000_000f64;
            let solver_name = format!("{:?}", solver);
            writeln!(
                perf_out,
                "{:>10} {:<12} {:>5} {:>10.0}",
                name, solver_name, cost, sol_time,
            )
            .unwrap();
        }
    };

    if opt.xml_instances {
        xml_instances(|name, p| solve_it(name, p));
    }
    if opt.txt_instances {
        txt_instances(|name, p| solve_it(name, p));
    }
    println!("{}", perf_out);
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
    pub fn testproblem() {
        let problem = crate::problem::problem1_with_stations();
        let result =
            ddd::solvers::maxsatddd::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                .unwrap()
                .0;
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

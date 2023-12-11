use serde::Serialize;
use std::{collections::HashMap, fmt::Display};

use super::{maxsatddd::SolveStats, SolverError};
use crate::{
    problem::{DelayCostType, Problem},
    solvers::bigm::visit_conflicts,
};

#[derive(Serialize)]
struct VarInfo {
    train: usize,
    visit: usize,
    time: i32,
}

struct Clause {
    weight: Option<u32>,
    lits: Vec<isize>,
}

#[derive(Default)]
struct WCNF {
    variables: Vec<VarInfo>,
    clauses: Vec<Clause>,
}

impl WCNF {
    pub fn new_var(&mut self, info: VarInfo) -> isize {
        self.variables.push(info);
        self.variables.len() as isize
    }

    pub fn add_clause(&mut self, weight: Option<u32>, lits: Vec<isize>) {
        if weight != Some(0) {
            self.clauses.push(Clause { weight, lits });
        }
    }
}

impl Display for WCNF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "c cnf file for partial weighted maxsat")?;

        for (var_idx, var) in self.variables.iter().enumerate() {
            writeln!(
                f,
                "c VAR {}",
                serde_json::to_string(&(var_idx, var)).unwrap()
            )?;
        }

        let hard_weight = self
            .clauses
            .iter()
            .map(|w| w.weight.unwrap_or(0))
            .sum::<u32>()
            + 1;

        writeln!(
            f,
            "p wcnf {} {} {}",
            self.variables.len(),
            self.clauses.len(),
            hard_weight
        )?;

        for c in self.clauses.iter() {
            write!(f, "{}", c.weight.unwrap_or(hard_weight))?;
            for l in c.lits.iter() {
                write!(f, " {}", *l)?;
            }
            writeln!(f, " 0")?;
        }

        Ok(())
    }
}

fn at_most_one(problem: &mut WCNF, set: &[isize]) {
    for i in 0..set.len() {
        for j in (i + 1)..set.len() {
            problem.add_clause(None, vec![-set[i], -set[j]]);
        }
    }
}

fn exactly_one(problem: &mut WCNF, set: &[isize]) {
    problem.add_clause(None, set.iter().map(|v| *v).collect());
    at_most_one(problem, set);
}

pub fn solve(
    env: &grb::Env,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
    discretization_interval: u32,
    big_m: u32,
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let start_time = std::time::Instant::now();
    let mut solver_time = std::time::Duration::ZERO;

    let mut conflicting_resources: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in problem.conflicts.iter() {
        conflicting_resources.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicting_resources.entry(*b).or_default().push(*a);
        }
    }

    let mut satproblem = WCNF::default();
    let mut resource_visits: Vec<Vec<(usize, usize)>> = Vec::new();

    let discretization_interval = discretization_interval as i32;
    let big_m = big_m as i32;
    let round = |t: i32| {
        ((t + discretization_interval / 2) / discretization_interval) * discretization_interval
    };

    let mut t_vars = Vec::new();

    // Discretize each visit's time point and compute the costs.
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_ts = Vec::new();
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push((train_idx, visit_idx));

            let mut time_vars = Vec::new();
            let mut time = round(visit.earliest);
            while time < visit.earliest + big_m {
                let v = satproblem.new_var(VarInfo {
                    train: train_idx,
                    visit: visit_idx,
                    time,
                });
                time_vars.push((time, v));
                let cost = train.visit_delay_cost(delay_cost_type, visit_idx, time) as u32;
                satproblem.add_clause(Some(cost), vec![-v]);
                time += discretization_interval;
            }
            train_ts.push(time_vars);
        }
        t_vars.push(train_ts);
    }

    println!("T_VARS");

    // CONSTRAINT 1: select one interval per time slot
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            exactly_one(
                &mut satproblem,
                &t_vars[train_idx][visit_idx]
                    .iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>(),
            );
        }
    }

    println!(
        "C1 vars={} clauses={}",
        satproblem.variables.len(),
        satproblem.clauses.len()
    );

    // CONSTRAINT 2: travel time constraints
    for (t_idx, train) in problem.trains.iter().enumerate() {
        for (v1_idx, visit) in train.visits.iter().enumerate() {
            if v1_idx + 1 < train.visits.len() {
                let v2_idx = v1_idx + 1;
                for (t1, v1) in t_vars[t_idx][v1_idx].iter() {
                    for (t2, v2) in t_vars[t_idx][v2_idx].iter() {
                        let can_reach = *t1 + visit.travel_time <= *t2;
                        // if t_idx == 0 && v1_idx == 0 {
                        //     println!("  t0v0 {}-{} -- {}", t1, t2, can_reach);
                        // }
                        if !can_reach {
                            satproblem.add_clause(None, vec![-v1, -v2]);
                        }
                    }
                }
            }
        }
    }

    println!(
        "C2 vars={} clauses={}",
        satproblem.variables.len(),
        satproblem.clauses.len()
    );

    // CONSTRAINT 3: resource conflict
    for (train1_idx, train) in problem.trains.iter().enumerate() {
        println!("   c3 t{}", train1_idx);
        for (visit1_idx, visit) in train.visits.iter().enumerate() {
            // Find conflicting visits
            if let Some(conflicting_resources) = conflicting_resources.get(&visit.resource_id) {
                for other_resource in conflicting_resources.iter().copied() {
                    for (train2_idx, visit2_idx) in resource_visits[other_resource].iter().copied()
                    {
                        if train2_idx == train1_idx {
                            continue;
                        }

                        for (t1_in, var1) in t_vars[train1_idx][visit1_idx].iter() {
                            for (t2_in, var2) in t_vars[train2_idx][visit2_idx].iter() {
                                let t1_out = t1_in + visit.travel_time;
                                let t2_out = t2_in
                                    + problem.trains[train2_idx].visits[visit2_idx].travel_time;

                                let separation = (*t2_in - t1_out).max(*t1_in - t2_out);
                                let has_separation = separation >= 0;

                                if !has_separation {
                                    satproblem.add_clause(None, vec![-var1, -var2]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "C3 vars={} clauses={}",
        satproblem.variables.len(),
        satproblem.clauses.len()
    );

    // std::fs::write("test.wcnf", format!("{}", satproblem)).unwrap();
    // todo!();

    // let output = std::fs::read_to_string("eval_out.txt").unwrap();
    // let mut input_vec = Vec::new();
    // for line in output.lines() {
    //     if line.starts_with("v ") {
    //         input_vec = line
    //             .chars()
    //             .filter_map(|v| {
    //                 if v == '0' {
    //                     Some(false)
    //                 } else if v == '1' {
    //                     Some(true)
    //                 } else {
    //                     None
    //                 }
    //             })
    //             .collect::<Vec<_>>();
    //     }
    // }

    let output = std::fs::read_to_string("uwr_out.txt").unwrap();
    let mut input_vec = Vec::new();
    for line in output.lines() {
        if line.starts_with("v ") {
            let mut counter = 0;
            input_vec = line
                .split(' ')
                .filter_map(|v| {
                    if v == "v" {
                        None
                    } else {
                        counter += 1;
                        if v.starts_with("-") {
                            assert!(v[1..].parse::<i32>().unwrap() == counter);
                            Some(false)
                        } else {
                            assert!(v.parse::<i32>().unwrap() == counter);
                            Some(true)
                        }
                    }
                })
                .collect::<Vec<_>>();
        }
    }

    println!("Got solution with {} vars", input_vec.len());
    assert!(input_vec.len() == satproblem.variables.len());

    let mut solution = Vec::new();
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_sol = Vec::new();
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            for (var_idx, value) in input_vec.iter().enumerate() {
                let var_info = &satproblem.variables[var_idx];
                if *value && var_info.train == train_idx && var_info.visit == visit_idx {
                    train_sol.push(var_info.time);
                }
            }

            // Workaround for avoiding conflict in the special relaxation where stations are uncapacitated:
            if visit.resource_id == 0 && visit_idx > 0 {
                train_sol[visit_idx] =
                    train_sol[visit_idx - 1] + train.visits[visit_idx - 1].travel_time;
            }

            assert!(train_sol.len() == visit_idx + 1);
        }

        train_sol.push(train_sol.last().unwrap() + train.visits.last().unwrap().travel_time);

        solution.push(train_sol);
    }

    let mut priorities = Vec::new();
    for (a @ (t1, v1), b @ (t2, v2)) in visit_conflicts(&problem) {
        if solution[t1][v1] <= solution[t2][v2] {
            priorities.push((a, b));
        } else {
            priorities.push((b, a));
        }
    }

    let sol = crate::solvers::minimize::minimize_solution(env, problem, priorities).unwrap();

    return Ok((sol, SolveStats::default()));

    todo!()
}

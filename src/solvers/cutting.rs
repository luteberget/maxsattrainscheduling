use std::collections::{HashMap, HashSet};

use ddd_problem::problem::{Problem, DelayCostType};

use super::SolverError;

mod highs {
    use std::ffi::c_void;

    pub struct Lp {
        ptr: *mut c_void,
    }

    impl Lp {
        pub fn new() -> Self {
            let ptr = unsafe { highs_sys::Highs_create() };
            Self { ptr }
        }

        pub fn solve(&mut self) -> Result<bool, ()> {
            let run = unsafe { highs_sys::Highs_run(self.ptr) };
            (run == highs_sys::kHighsStatusOk).then_some(()).ok_or(())?;

            #[allow(clippy::match_like_matches_macro)]
            let status = match unsafe { highs_sys::Highs_getModelStatus(self.ptr) } {
                highs_sys::kHighsModelStatusOptimal => true,
                highs_sys::kHighsModelStatusModelEmpty => true,
                _ => false,
            };

            Ok(status)
        }

        pub fn get_solution(&self, vec: &mut Vec<f64>) -> Result<(), ()> {
            let num_cols = unsafe { highs_sys::Highs_getNumCols(self.ptr) } as usize;
            vec.resize(num_cols, f64::NAN);
            if num_cols == 0 {
                return Ok(());
            }

            let null = std::ptr::null_mut();
            let ptr = vec.as_mut_ptr();
            let res = unsafe { highs_sys::Highs_getSolution(self.ptr, ptr, null, null, null) };
            (res == highs_sys::kHighsStatusOk).then_some(()).ok_or(())?;
            Ok(())
        }

        pub fn add_var(&mut self, cost: f64, lb: f64, ub: f64) -> Result<(), ()> {
            let res = unsafe {
                highs_sys::Highs_addCol(
                    self.ptr,
                    cost,
                    lb,
                    ub,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            };
            (res == highs_sys::kHighsStatusOk).then_some(()).ok_or(())?;
            Ok(())
        }
    }
}

pub fn solve_cutting(
    problem: &Problem,
    delay_cost_type: DelayCostType,
    timeout: f64,
    train_names: &[String],
    resource_names: &[String],
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<Vec<Vec<i32>>, SolverError> {
    let visit_conflicts = crate::solvers::bigm::visit_conflicts(problem);
    let mut solver = highs::Lp::new();

    enum Var {
        Decision { name: String },
        Timing { train: usize, visit: usize },
    }

    // Debug info
    let mut priority_vars = HashMap::new();
    let mut added_conflicts = HashSet::new();
    // let mut visit_cont_vars = HashMap::new();

    let mut vars: Vec<Var> = Vec::new();
    let mut lp_incumbent: Vec<f64> = Vec::new();
    let mut solution_incumbent: Vec<Vec<i32>> = problem
        .trains
        .iter()
        .map(|t| t.visits.iter().map(|v| v.earliest).collect())
        .collect();

    'conflicts: loop {
        'integrality: loop {
            // Solve the LP
            if !solver.solve().expect("highs failed") {
                panic!("lp not optimal");
            }
            println!("lp ok");

            solver.get_solution(&mut lp_incumbent).unwrap();
            assert!(vars.len() == lp_incumbent.len());

            for (var_idx, (var, value)) in vars.iter().zip(lp_incumbent.iter()).enumerate() {
                if let Var::Decision { .. } = var {
                    if (value.round() - value).abs() > 1e-5 {
                        println!("var {} is fractional", var_idx);
                        todo!("cuts");
                        continue 'integrality;
                    }
                }
            }
            break;
        }

        let violated_conflicts = {
            let _p = hprof::enter("check conflicts");

            let mut cs: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
            for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
                if !crate::solvers::bigm::check_conflict(visit_pair, |t, v| {
                    solution_incumbent[t][v]
                })? {
                    cs.entry((t1, t2)).or_default().push((v1, v2));
                }
            }
            cs
        };

        if !violated_conflicts.is_empty() {
            for ((t1, t2), pairs) in violated_conflicts {
                let (v1, v2) = *pairs.iter().min_by_key(|(v1, v2)| v1 + v2).unwrap();
                let visit_pair = ((t1, v1), (t2, v2));
                // let conflict = ConflictInformation {
                //     problem,
                //     visit_pair,
                //     t_vars: &t_vars,
                //     train_names,
                //     resource_names,
                // };
                let var_idx = vars.len();
                let choice_var_name = format!(
                    "confl_tn{}_v{}_tn{}_v{}",
                    train_names[t1], v1, train_names[t2], v2
                );
                vars.push(Var::Decision {
                    name: choice_var_name,
                });
                // let var = add_conflict_constraint(&mut model, &conflict)?;
                assert!(priority_vars.insert(visit_pair, var_idx).is_none());
                assert!(added_conflicts.insert(visit_pair));
            }
        } else {
            return Ok(solution_incumbent);
        }
    }
}

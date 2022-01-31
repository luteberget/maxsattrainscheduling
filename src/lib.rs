use satcoder::{Bool, SatResultWithCore, SatSolverWithCore};

pub mod debug;
pub mod parser;
pub mod problem;
pub mod solvers;

pub fn minimize_core<L: satcoder::Lit>(
    core: &mut Vec<Bool<L>>,
    solver: &mut impl SatSolverWithCore<Lit = L>,
) {
    // println!("Starting core minimization.");
    let mut printed = false;
    let mut i = 0;
    'minimize_loop: loop {
        for _ in 0..core.len() {
            let last_core_size = core.len();
            let mut assumptions = core.clone();
            let remove_idx = i % assumptions.len();
            assumptions.remove(remove_idx);
            // println!(
            //     "Solving core #{}->{} removed {}",
            //     core.len(),
            //     assumptions.len(),
            //     remove_idx
            // );
            let result = solver.solve_with_assumptions(assumptions.iter().copied());
            if let SatResultWithCore::Unsat(new_core) = result {
                printed = true;
                print!("  mz {}->{}", last_core_size, new_core.len());
                *core = new_core.iter().map(|c| Bool::Lit(*c)).collect();
                continue 'minimize_loop;
            }
            i += 1;
        }
        break;
    }
    if printed {
        println!();
    }
}

pub fn trim_core<L: satcoder::Lit>(
    core: &mut Vec<Bool<L>>,
    solver: &mut impl SatSolverWithCore<Lit = L>,
) {
    let mut trimmed = false;
    // println!("Starting core trim.");
    loop {
        let last_core_size = core.len();
        // Try to trim the core.
        let result = solver.solve_with_assumptions(core.iter().copied());
        if let SatResultWithCore::Unsat(new_core) = result {
            if new_core.len() < last_core_size {
                trimmed = true;
                print!("  tr {}->{}", last_core_size, new_core.len());
                *core = new_core.iter().map(|c| Bool::Lit(*c)).collect();
            } else {
                break;
            }
        } else {
            panic!("Should not become SAT when calling the solver again.");
        }
    }
    if trimmed {
        println!();
    }
}

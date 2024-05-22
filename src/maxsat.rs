
#[derive(Clone, Copy)]
pub enum MaxSatError {
    NoSolution,
    Timeout,
}

pub trait MaxSatSolver {
    fn add_clause(&mut self, weight: Option<u32>, clause: Vec<isize>);
    fn new_var(&mut self) -> isize;
    fn status(&mut self) -> String;
    fn optimize(
        &mut self,
        timeout: Option<f64>,
        assumptions: impl Iterator<Item = isize>,
    ) -> Result<(i32, Vec<bool>), MaxSatError>;

    fn at_most_one(&mut self, set: &[isize]) {
        if set.len() <= 5 {
            for i in 0..set.len() {
                for j in (i + 1)..set.len() {
                    self.add_clause(None, vec![-set[i], -set[j]]);
                }
            }
        } else {
            let new_var = self.new_var();
            let (s1, s2) = set.split_at(set.len() / 2);
            let mut s1 = s1.to_vec();
            let mut s2 = s2.to_vec();
            s1.push(new_var);
            s2.push(-new_var);
            self.at_most_one(&s1);
            self.at_most_one(&s2);
        }
    }

    fn exactly_one(&mut self, set: &[isize]) {
        self.add_clause(None, set.iter().map(|v| *v).collect::<Vec<_>>());
        self.at_most_one(set);
    }
}

pub struct IPAMIRSolver {
    ipamir: ipamir_rs::IPAMIR,
    n_vars: usize,
    n_clauses: usize,
}

impl std::fmt::Debug for IPAMIRSolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Incremental")
            .field("ipamir", &self.ipamir.signature())
            .field("n_vars", &self.n_vars)
            .field("n_clauses", &self.n_clauses)
            .finish()
    }
}

impl IPAMIRSolver {
    pub fn new() -> Self {
        Self {
            ipamir: ipamir_rs::IPAMIR::new(),
            n_clauses: 0,
            n_vars: 0,
        }
    }
}

impl MaxSatSolver for IPAMIRSolver {
    fn add_clause(&mut self, weight: Option<u32>, clause: Vec<isize>) {
        if let Some(w) = weight {
            if clause.len() != 1 {
                panic!("only soft lits supported (not clauses)");
            }
            self.ipamir.add_soft_lit(-clause[0] as i32, w as u64);
        } else {
            self.ipamir.add_clause(clause.iter().map(|l| *l as i32));
            self.n_clauses += 1;
        }
    }

    fn new_var(&mut self) -> isize {
        self.n_vars += 1;
        self.n_vars as isize
    }

    fn status(&mut self) -> String {
        format!(
            "IPAMIR={} vars={} clauses={}",
            self.ipamir.signature(),
            self.n_vars,
            self.n_clauses
        )
    }

    fn optimize(
        &mut self,
        timeout: Option<f64>,
        assumptions: impl Iterator<Item = isize>,
    ) -> Result<(i32, Vec<bool>), MaxSatError> {
        if timeout.map(|x| x <= 0.0).unwrap_or(false) {
            return Err(MaxSatError::Timeout);
        }

        match {
            self.ipamir.solve(
                timeout.map(|t| std::time::Duration::from_secs_f64(t)),
                assumptions.map(|l| l as i32),
            )
        } {
            ipamir_rs::MaxSatResult::Optimal(s) => {
                let obj = s.get_objective_value() as i32;
                let lits = (0..self.n_vars).map(|i| (i + 1) as i32);
                let values = lits.map(|l| s.get_literal_value(l) == l);
                Ok((obj, values.collect()))
            }
            ipamir_rs::MaxSatResult::Unsat => Err(MaxSatError::NoSolution),
            ipamir_rs::MaxSatResult::Error => panic!(),
            ipamir_rs::MaxSatResult::Timeout(_) => Err(MaxSatError::Timeout),
        }
    }
}
use std::io::{BufRead, BufReader};

#[derive(Clone, Copy)]
pub enum MaxSatError {
    NoSolution,
    Timeout,
}

pub trait MaxSatSolver {
    fn add_clause(&mut self, weight: Option<u32>, clause: Vec<isize>);
    fn new_var(&mut self) -> isize;
    fn status(&mut self) -> String;
    fn optimize(&mut self, timeout: Option<f64>) -> Result<(i32, Vec<bool>), MaxSatError>;

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

struct Clause {
    weight: Option<u32>,
    lits: Vec<isize>,
}

#[derive(Default)]
pub struct WCNF {
    variables: Vec<()>,
    clauses: Vec<Clause>,
}

impl WCNF {
    pub fn new_var(&mut self) -> isize {
        self.variables.push(());
        self.variables.len() as isize
    }

    pub fn add_clause(&mut self, weight: Option<u32>, lits: Vec<isize>) {
        if weight != Some(0) {
            self.clauses.push(Clause { weight, lits });
        }
    }
}

impl std::fmt::Display for WCNF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "c cnf file for partial weighted maxsat")?;

        // for (var_idx, var) in self.variables.iter().enumerate() {
        //     writeln!(
        //         f,
        //         "c VAR {}",
        //         serde_json::to_string(&(var_idx, var)).unwrap()
        //     )?;
        // }

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

pub struct Incremental {
    ipamir: ipamir_rs::IPAMIR,
    n_vars: usize,
    n_clauses: usize,
}

impl Incremental {
    pub fn new() -> Self {
        Self {
            ipamir: ipamir_rs::IPAMIR::new(),
            n_clauses: 0,
            n_vars: 0,
        }
    }
}

impl MaxSatSolver for Incremental {
    fn add_clause(&mut self, weight: Option<u32>, clause: Vec<isize>) {
        if let Some(w) = weight {
            if clause.len() != 1 {
                panic!("only soft lits supported (not clauses)");
            }
            self.ipamir.add_soft_lit(clause[0] as i32, w as u64);
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

    fn optimize(&mut self, timeout: Option<f64>) -> Result<(i32, Vec<bool>), MaxSatError> {
        match self.ipamir.solve(
            timeout.map(|t| std::time::Duration::from_secs_f64(t)),
            std::iter::empty(),
        ) {
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

pub struct External {
    // filename: String,
    wcnf: WCNF,
}

impl External {
    pub fn new() -> Self {
        Self {
            // filename: filename.to_string(),
            wcnf: Default::default(),
        }
    }
}

impl MaxSatSolver for External {
    fn add_clause(&mut self, weight: Option<u32>, clause: Vec<isize>) {
        self.wcnf.add_clause(weight, clause)
    }

    fn new_var(&mut self) -> isize {
        self.wcnf.new_var()
    }

    fn optimize(&mut self, timeout: Option<f64>) -> Result<(i32, Vec<bool>), MaxSatError> {
        let _p = hprof::enter("external maxsat solver");
        {
            let _p1: hprof::ProfileGuard<'_> = hprof::enter("write wcnf");
            std::fs::write("temp.wcnf", format!("{}", &self.wcnf)).unwrap();
        }

        println!("Running external solver");

        let _p2: hprof::ProfileGuard<'_> = hprof::enter("run external");

        let cmd = if let Some(time) = timeout {
            duct::cmd!("timeout", &format!("{}", time), "./uwrmaxsat", "temp.wcnf")
        } else {
            duct::cmd!("./uwrmaxsat", "temp.wcnf")
        };
        let reader = cmd.stderr_to_stdout().reader().unwrap();
        let mut output = String::new();
        let lines = BufReader::with_capacity(10, reader).lines();
        for line in lines {
            if let Ok(line) = line {
                // println!("LINE {}", line);
                output.push_str(&line);
                output.push('\n');
            } else {
                return Err(MaxSatError::Timeout);
            }
        }
        drop(_p2);
        let _p3 = hprof::enter("read solution");

        // let cmd = duct::cmd!("echo", "temp.wcnf");
        // // let k = cmd.stdout_capture().read().unwrap();
        // // println!("KK {}", k);
        // let mut reader = cmd.stdout_capture().reader().unwrap();
        // let mut buf = Vec::new();
        // // let mut reader = BufReader::new(reader);
        // let mut output = String::new();
        // println!("tryin gto read.");
        // while let Ok(xx) = reader.read(&mut buf) {
        //     println!("Read {} bytes", xx);
        //     let x = String::from_utf8_lossy(&buf);
        //     print!("{}", x);
        //     output.extend(x.chars());
        //     buf.clear();
        //     if xx == 0 {
        //         std::thread::sleep(std::time::Duration::from_millis(50));
        //     }
        // }

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

        // let output = std::fs::read_to_string("uwr_out.txt").unwrap();
        let mut input_vec = Vec::new();
        let mut obj_val = i32::MIN;
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
            } else if line.starts_with("o ") {
                obj_val = line.split(' ').nth(1).unwrap().parse::<i32>().unwrap();
            }
        }

        println!("Got solution with {} vars", input_vec.len());
        assert!(input_vec.len() == self.wcnf.variables.len());

        Ok((obj_val, input_vec))
    }

    fn status(&mut self) -> String {
        format!(
            "External solver vars={} clauses={}",
            self.wcnf.variables.len(),
            self.wcnf.clauses.len()
        )
    }
}

use std::{
    ffi::{c_char, c_int, c_void, CStr},
    ptr::null,
    time::{Duration, Instant},
};


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
    ipamir: IPAMIRExternSolver,
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
            ipamir: IPAMIRExternSolver::new(),
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
                timeout.filter(|t| t.is_finite()).map(|t| std::time::Duration::from_secs_f64(t)),
                assumptions.map(|l| l as i32),
            )
        } {
            MaxSatResult::Optimal(s) => {
                let obj = s.get_objective_value() as i32;
                let lits = (0..self.n_vars).map(|i| (i + 1) as i32);
                let values = lits.map(|l| s.get_literal_value(l) == l);
                Ok((obj, values.collect()))
            }
            MaxSatResult::Unsat => Err(MaxSatError::NoSolution),
            MaxSatResult::Error => panic!(),
            MaxSatResult::Timeout(_) => Err(MaxSatError::Timeout),
        }
    }
}

extern "C" {
    fn ipamir_signature() -> *const c_char;
    fn ipamir_init() -> *const c_void;
    fn ipamir_release(solver: *const c_void);
    fn ipamir_add_hard(solver: *const c_void, lit_or_zero: i32);
    fn ipamir_add_soft_lit(solver: *const c_void, lit: i32, weight: u64);
    fn ipamir_assume(solver: *const c_void, lit: i32);
    fn ipamir_solve(solver: *const c_void) -> c_int;
    fn ipamir_val_obj(solver: *const c_void) -> u64;
    fn ipamir_val_lit(solver: *const c_void, lit: i32) -> i32;
    fn ipamir_set_terminate(
        solver: *const c_void,
        state: *const c_void,
        x: Option<extern "C" fn(state: *const c_void) -> c_int>,
    );
}

pub struct Solution<'a> {
    ipamir: &'a mut IPAMIRExternSolver,
}

pub enum MaxSatResult<'a> {
    Timeout(Option<Solution<'a>>),
    Optimal(Solution<'a>),
    Unsat,
    Error,
}

pub struct IPAMIRExternSolver {
    ptr: *const c_void,
}

impl IPAMIRExternSolver {
    pub fn new() -> Self {
        let ptr = unsafe { ipamir_init() };
        assert!(ptr != null());
        IPAMIRExternSolver { ptr }
    }

    pub fn signature(&self) -> &str {
        let c_buf: *const c_char = unsafe { ipamir_signature() };
        let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
        let str_slice: &str = c_str.to_str().unwrap();
        str_slice
    }

    pub fn add_soft_lit(&mut self, lit: i32, weight: u64) {
        unsafe { ipamir_add_soft_lit(self.ptr, lit, weight) };
    }

    pub fn add_clause(&mut self, lits: impl Iterator<Item = i32>) {
        for lit in lits {
            unsafe { ipamir_add_hard(self.ptr, lit) };
        }
        unsafe { ipamir_add_hard(self.ptr, 0) };
    }

    pub fn solve(
        &mut self,
        timeout: Option<Duration>,
        assumptions: impl Iterator<Item = i32>,
    ) -> MaxSatResult {
        for lit in assumptions {
            unsafe { ipamir_assume(self.ptr, lit) };
        }

        struct CallbackUserData {
            start_time: Instant,
            timeout: Duration,
        }
        let mut userdata: Option<CallbackUserData> = None;

        if let Some(timeout) = timeout {
            userdata = Some(CallbackUserData {
                start_time: Instant::now(),
                timeout,
            });

            extern "C" fn cb(state: *const c_void) -> c_int {
                let ptr = state as *const CallbackUserData;
                let user_data = unsafe { &*ptr };

                if user_data.start_time.elapsed() > user_data.timeout {
                    1
                } else {
                    0
                }
            }

            unsafe {
                ipamir_set_terminate(
                    self.ptr,
                    userdata.as_ref().unwrap() as *const CallbackUserData as *const c_void,
                    Some(cb),
                )
            }
        }

        let code = unsafe { ipamir_solve(self.ptr) };

        if userdata.is_some() {
            unsafe { ipamir_set_terminate(self.ptr, null(), None) };
        }

        if code == 0 {
            MaxSatResult::Timeout(None)
        } else if code == 10 {
            MaxSatResult::Timeout(Some(Solution { ipamir: self }))
        } else if code == 20 {
            MaxSatResult::Unsat
        } else if code == 30 {
            MaxSatResult::Optimal(Solution { ipamir: self })
        } else if code == 40 {
            MaxSatResult::Error
        } else {
            panic!("unrecogized return code from ipamir");
        }
    }
}

impl Drop for IPAMIRExternSolver {
    fn drop(&mut self) {
        unsafe { ipamir_release(self.ptr) };
    }
}

impl<'a> Solution<'a> {
    pub fn get_objective_value(&self) -> u64 {
        unsafe { ipamir_val_obj(self.ipamir.ptr) }
    }

    pub fn get_literal_value(&self, lit: i32) -> i32 {
        unsafe { ipamir_val_lit(self.ipamir.ptr, lit) }
    }
}

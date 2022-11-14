use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use tinyvec::TinyVec;

use crate::{
    problem::*,
    singletrain::{onetrain, Constraint, SingleTrainSolver, TimeInterval},
};

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    SolveNode,
}

pub struct ConflictSolverNode {

}

pub struct ConflictSolver {
    pub problem: Rc<Problem>,
    pub trains: Vec<SingleTrainSolver>,
    pub conflicts: Vec<TrackConflicts>,

    pub open :Vec<Rc<ConflictSolverNode>>,
    pub current_node :Option<Rc<ConflictSolverNode>>,
    pub closed :Vec<Rc<ConflictSolverNode>>,
}

#[derive(Default)]
pub struct Conflict {}

pub struct TrackConflicts {
    pub conflicts: TinyVec<[Conflict; 4]>,
}

impl ConflictSolver {

    pub fn status(&self) -> ConflictSolverStatus {
        match (self.open.len() > 0, self.current_node.is_some()) {
            (_, true) => ConflictSolverStatus::SolveNode,
            (true, _) => ConflictSolverStatus::SelectNode,
            (false, false) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn new(problem: Rc<Problem>) -> Self {
        let trains = problem
            .trains
            .iter()
            .map(|_t| SingleTrainSolver::new())
            .collect();

            
        let conflicts = problem
            .tracks
            .iter()
            .map(|_t| TrackConflicts {
                conflicts: TinyVec::new(),
            })
            .collect();

        let start_node = ConflictSolverNode {

        };



        Self {
            problem,
            trains,
            conflicts,
            closed: Default::default(),
            current_node: Default::default(),
            open: vec![Rc::new(start_node)],
        }
    }
}

pub fn solve(problem: &Problem) {
    let start: BTreeMap<i32, BTreeMap<i32, Vec<TimeInterval>>> = Default::default();
    let empty: BTreeMap<i32, Vec<TimeInterval>> = Default::default();

    if let Some((path, cost)) = pathfinding::directed::astar::astar(
        &start,
        |constraints| {
            let solutions = problem
                .trains
                .iter()
                .enumerate()
                .map(|(tid, t)| {
                    onetrain(
                        t,
                        &problem.tracks,
                        constraints.get(&(tid as _)).unwrap_or(&empty),
                    )
                })
                .collect::<Vec<_>>();
            let x = find_conflict(&solutions);

            let c = constraints.clone();
            x.into_iter().flat_map(move |(a, b)| {
                std::iter::once((add_conflict(&c, a), 0))
                    .chain(std::iter::once((add_conflict(&c, b), 0)))
            })
        },
        |n| 0,
        |n| {
            let solutions = problem
                .trains
                .iter()
                .enumerate()
                .map(|(tid, t)| onetrain(t, &problem.tracks, n.get(&(tid as _)).unwrap_or(&empty)))
                .collect::<Vec<_>>();
            let x = find_conflict(&solutions);
            x.is_none()
        },
    ) {
        println!("Cost {}", cost);
    } else {
        panic!();
    }
}

pub fn add_conflict(
    conflicts: &BTreeMap<i32, BTreeMap<i32, Vec<TimeInterval>>>,
    (train, track, interval): Constraint,
) -> BTreeMap<i32, BTreeMap<i32, Vec<TimeInterval>>> {
    let mut conflicts = conflicts.clone();
    let vec = conflicts
        .entry(train)
        .or_default()
        .entry(track)
        .or_default();
    vec.push(interval);
    vec.sort_by_key(|k| k.time_start);
    conflicts
}

pub fn find_conflict(
    solutions: &Vec<Vec<(TrackRef, TimeValue)>>,
) -> Option<(Constraint, Constraint)> {
    todo!()
}

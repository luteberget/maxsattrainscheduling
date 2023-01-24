use log::debug;

use crate::{
    branching::ConflictConstraint,
    occupation::ResourceConflicts,
    problem::{Problem, TimeValue, TrainRef},
    TrainSolver, TrainSolverStatus,
};

pub struct TrainSet<Train> {
    pub trains: Vec<Train>,
    pub slacks: Vec<Vec<i32>>,
    pub train_lbs: Vec<i32>,
    pub train_const_lbs: Vec<i32>,
    pub lb: i32,
    pub dirty_trains: Vec<u32>,
    pub original_trains: Vec<crate::problem::Train>,
}

impl<Train: TrainSolver> TrainSet<Train> {
    pub fn is_dirty(&self) -> bool {
        !self.dirty_trains.is_empty()
    }

    pub fn solve_all_trains(&mut self, conflicts: &mut ResourceConflicts) {
        while let Some(&dirty_train) = self.dirty_trains.last() {
            let x = self.solve_train_step(&dirty_train, conflicts);
            if x == Some(false) {
                println!("Train failed {}", dirty_train);
                break;
            }
        }
    }

    pub fn solve_first_train(&mut self, conflicts: &mut ResourceConflicts) {
        let train = *self.dirty_trains.last().unwrap();
        loop {
            if self.solve_train_step(&train, conflicts).is_some() {
                break;
            }
        }
    }

    pub fn solve_step(&mut self, conflicts: &mut ResourceConflicts) -> Option<bool> {
        let train = *self.dirty_trains.last().unwrap();
        self.solve_train_step(&train, conflicts)
    }

    pub fn current_solution(&self) -> (i32, Vec<Vec<TimeValue>>) {
        assert!(self.dirty_trains.is_empty());
        let mut total_cost = 0;
        let mut out_vec = Vec::new();
        for (cost, times) in self.trains.iter().map(|t| t.current_solution()) {
            total_cost += cost;
            out_vec.push(times);
        }

        (total_cost, out_vec)
    }

    pub fn add_remove_constraint(
        &mut self,
        add: bool,
        constraint: &ConflictConstraint,
        conflicts: &mut ResourceConflicts,
    ) {
        {
            if add {
                let train = constraint.train;
                self.trains[train as usize].set_occupied(
                    true,
                    constraint.resource,
                    constraint.enter_after,
                    constraint.exit_before,
                    &mut |a, block, res, i| conflicts.add_or_remove(a, train, block, res, i),
                );

                self.lb -= self.train_lbs[train as usize];
                self.train_lbs[train as usize] = 0;
                self.queue_rewinded_train(train);
            } else {
                let train = constraint.train;

                // Remove constraint
                self.trains[train as usize].set_occupied(
                    false,
                    constraint.resource,
                    constraint.enter_after,
                    constraint.exit_before,
                    &mut |a, b, r, i| conflicts.add_or_remove(a, train, b, r, i),
                );

                self.lb -= self.train_lbs[train as usize];
                self.train_lbs[train as usize] = 0;

                self.queue_rewinded_train(train);
            }
        }
    }

    fn solve_train_step(
        &mut self,
        dirty_train: &u32,
        conflicts: &mut ResourceConflicts,
    ) -> Option<bool> {
        let dirty_train_idx = *dirty_train as usize;
        match self.trains[dirty_train_idx].status() {
            TrainSolverStatus::Failed => {
                assert!(self.train_lbs[dirty_train_idx] == 0);
                debug!("Train {} failed.", dirty_train_idx);
                Some(false)
            }
            TrainSolverStatus::Optimal => {
                let prev_cost = self.train_lbs[dirty_train_idx];
                let train_cost = self.trains[dirty_train_idx].current_solution().0;
                assert!(train_cost >= self.train_const_lbs[dirty_train_idx]);
                let new_cost = train_cost - self.train_const_lbs[dirty_train_idx];
                self.train_lbs[dirty_train_idx] = new_cost;
                self.lb += new_cost - prev_cost;

                debug!(
                    "Train {} optimal. train_LB={} problem_LB={}",
                    dirty_train_idx, new_cost, self.lb
                );
                self.dirty_trains.retain(|x| x != dirty_train);
                Some(true)
            }
            TrainSolverStatus::Working => {
                assert!(self.train_lbs[dirty_train_idx] == 0);

                self.trains[dirty_train_idx].step(&mut |add, block, resource, interval| {
                    conflicts.add_or_remove(
                        add,
                        dirty_train_idx as TrainRef,
                        block,
                        resource,
                        interval,
                    )
                });

                self.train_queue_bubble_leftward();
                None
            }
        }
    }

    fn train_queue_bubble_leftward(&mut self) {
        // Bubble the last train in the dirty train queue to the correct ordering.
        let mut idx = self.dirty_trains.len();
        while idx >= 2
            && self.trains[self.dirty_trains[idx - 2] as usize].order_time()
                < self.trains[self.dirty_trains[idx - 1] as usize].order_time()
        {
            self.dirty_trains.swap(idx - 2, idx - 1);
            idx -= 1;
        }
    }

    pub fn queue_rewinded_train(&mut self, train: TrainRef) {
        if let Some((mut idx, _)) = self.dirty_trains
            .iter()
            .enumerate()
            .rev() /* Search from the back, because a conflicting train is probably recently used. */
            .find(|(_, t)| **t == train as u32)
        {
            // Bubble to the right, in this case, because the train has become earlier.
            while idx + 1 < self.dirty_trains.len()
                && self.trains[self.dirty_trains[idx] as usize].order_time()
                    < self.trains[self.dirty_trains[idx+1] as usize].order_time()
            {
                self.dirty_trains.swap(idx, idx +1);
                idx += 1;
            }
        } else {
            self.dirty_trains.push(train as u32);
            self.train_queue_bubble_leftward();
        }
    }

    pub fn new(problem: Problem) -> Self {
        let original_trains = problem.trains.clone();

        let slacks = problem
            .trains
            .iter()
            .map(|train| {
                let mut slacks: Vec<i32> = train
                    .blocks
                    .iter()
                    .map(|b| b.delayed_after.unwrap_or(999999i32))
                    .collect();
                for (block_idx, block) in train.blocks.iter().enumerate().rev() {
                    if slacks[block_idx] == 999999 {
                        slacks[block_idx] = block.minimum_travel_time
                            + block
                                .nexts
                                .iter()
                                .map(|n| slacks[*n as usize])
                                .min()
                                .unwrap_or(999999i32);
                    }
                }
                slacks
            })
            .collect();

        let train_const_lbs: Vec<i32> = problem
            .trains
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut t = Train::new(i, t.clone());
                while !matches!(t.status(), TrainSolverStatus::Optimal) {
                    t.step(&mut |_, _, _, _| {});
                }
                t.current_solution().0
            })
            .collect();

        println!("Train LBs {:?}", train_const_lbs);

        let trains: Vec<Train> = problem
            .trains
            .into_iter()
            .enumerate()
            .map(|(i, t)| Train::new(i, t))
            .collect();

        let mut dirty_trains: Vec<u32> = (0..(trains.len() as u32)).collect();
        dirty_trains.sort_by_key(|t| -(trains[*t as usize].order_time() as i64));
        println!("train times {:?}", trains.iter().map(|t| t.order_time()).collect::<Vec<_>>());

        let train_lbs = trains.iter().map(|_| 0).collect();

        let trainset = TrainSet {
            original_trains,
            trains,
            slacks,
            lb: train_const_lbs.iter().sum(),
            train_lbs,
            train_const_lbs,
            dirty_trains,
        };
        trainset
    }
}

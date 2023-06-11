use crate::problem::{DelayCostType, Problem};
use super::{bigm::visit_conflicts, minimize, SolverError};

pub struct ChoiceCharacteristics {
    pub time: i32,
    pub offset: i32,
    pub objective_cost: i32,
    pub total_lateness: i32,
}

pub fn default_heuristic(a: &ChoiceCharacteristics, _b: &ChoiceCharacteristics) -> f64 {
    a.objective_cost as f64
}

pub fn solve2(
    problem: &Problem,
    env: &grb::Env,
    delay_cost_type: DelayCostType,
    heuristic_cost: impl Fn(&ChoiceCharacteristics, &ChoiceCharacteristics) -> f64,
) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("greedy solver");
    let visit_conflicts = visit_conflicts(problem);
    let mut priorities = Vec::new();
    'refinement: loop {
        let solution = {
            minimize::minimize_solution(env, problem, priorities.clone())?
        };
        let choices = {
            let _p = hprof::enter("check conflicts");

            let mut cs = Vec::new();
            for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
                if let Err((t1_delta, t2_delta)) =
                    check_conflict(visit_pair, |t, v| solution[t][v])?
                {
                    let pair1 = ((t1, v1), (t2, v2));
                    let pair2 = ((t2, v2), (t1, v1));
                    let priorities1 = priorities
                        .iter()
                        .copied()
                        .chain(std::iter::once(pair1))
                        .collect();
                    let priorities2 = priorities
                        .iter()
                        .copied()
                        .chain(std::iter::once(pair2))
                        .collect();
                    let solution1 = minimize::minimize_solution(env, problem, priorities1);
                    let solution2 = minimize::minimize_solution(env, problem, priorities2);

                    if let (Ok(solution1), Ok(solution2)) = (&solution1, &solution2) {
                        let objective_cost1 = problem.cost(solution1, delay_cost_type);
                        let total_lateness1 = (0..problem.trains.len())
                            .flat_map(move |t| {
                                (0..problem.trains[t].visits.len()).map(move |v| solution1[t][v])
                            })
                            .sum::<i32>();
                        let objective_cost2 = problem.cost(solution2, delay_cost_type);
                        let total_lateness2 = (0..problem.trains.len())
                            .flat_map(move |t| {
                                (0..problem.trains[t].visits.len()).map(move |v| solution2[t][v])
                            })
                            .sum::<i32>();

                        let choice1 = ChoiceCharacteristics {
                            time: solution[t1][v1],
                            offset: t1_delta,
                            objective_cost: objective_cost1,
                            total_lateness: total_lateness1,
                        };

                        let choice2 = ChoiceCharacteristics {
                            time: solution[t2][v2],
                            offset: t2_delta,
                            objective_cost: objective_cost2,
                            total_lateness: total_lateness2,
                        };

                        cs.push((heuristic_cost(&choice1, &choice2), pair1));
                        cs.push((heuristic_cost(&choice2, &choice1), pair2));
                    } else if let Ok(_solution1) = solution1 {
                        priorities.push(pair1);
                        continue 'refinement;
                    } else if let Ok(_solution2) = solution2 {
                        priorities.push(pair2);
                        continue 'refinement;
                    } else {
                        return Err(SolverError::NoSolution);
                    }
                }
            }
            cs
        };

        if let Some((_, choice)) = choices
            .iter()
            .min_by_key(|(cost, _)| (100000000.0 * *cost) as i64)
        {
            priorities.push(*choice);
        } else {
            return Ok(solution);
        }
    }
}

fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    t_vars: impl Fn(usize, usize) -> i32,
) -> Result<Result<i32, (i32, i32)>, SolverError> {
    let t_t1v1 = t_vars(t1, v1);
    let t_t1v1next = t_vars(t1, v1 + 1);
    let t_t2v2 = t_vars(t2, v2);
    let t_t2v2next = t_vars(t2, v2 + 1);

    let separation = (t_t2v2 - t_t1v1next).max(t_t1v1 - t_t2v2next);

    // let t1_first = t_t1v1next <= t_t2v2;
    // let t2_first = t_t2v2next <= t_t1v1;
    let has_separation = separation >= 0;
    // println!("t1v1 {}-{}  t2v2 {}-{}  separation {} is_separated {}", t_t1v1, t_t1v1next, t_t2v2, t_t2v2next, separation, has_separation);
    // assert!((separation >= 1e-5) == has_separation);

    if has_separation {
        Ok(Ok(separation))
    } else {
        // println!("conflict separation {:?}  {}", ((t1, v1), (t2, v2)), separation);

        let postpone_t1_delta = t_t2v2next - t_t1v1;
        let postpone_t2_delta = t_t1v1next - t_t2v2;

        Ok(Err((postpone_t1_delta, postpone_t2_delta)))
    }
}

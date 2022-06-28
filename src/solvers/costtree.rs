use std::collections::HashMap;

use satcoder::Bool;

#[derive(Debug)]
pub struct CostTree<L: satcoder::Lit> {
    nodes: Vec<CostTreeNode<L>>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
struct CostTreeNode<L: satcoder::Lit> {
    cost: usize,
    parent: Option<usize>,
    literal: satcoder::Bool<L>,
}

impl<L: satcoder::Lit + std::fmt::Debug> CostTree<L> {
    pub fn new() -> Self {
        CostTree { nodes: Vec::new() }
    }

    pub fn add_cost(
        &mut self,
        solver: &mut impl satcoder::SatInstance<L>,
        input: Bool<L>,
        cost: usize,
        // name: String,
        notify_vars: &mut impl FnMut(usize, Bool<L>),
    ) {
        assert!(cost > 0);

        // println!("Add cost {} to tree {:?}", cost, self.nodes);

        if !self.nodes.iter().any(|n| n.cost == cost) {
            // Add a new node.

            // Find the next lowest cost.
            let parent_node_idx = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.cost < cost)
                .max_by_key(|(_, n)| n.cost)
                .map(|(i, _n)| i);

            let dist_prev = cost - parent_node_idx.map(|i| self.nodes[i].cost).unwrap_or(0);
            let linear_mode = dist_prev < 10;
            // let linear_mode = false;

            if linear_mode && parent_node_idx.map(|i| self.nodes[i].cost).unwrap_or(0) + 1 != cost {
                self.add_cost(solver, false.into(), cost - 1,  notify_vars);
                return self.add_cost(solver, input, cost,  notify_vars);
            } else {
                let new_var = solver.new_var();

                let (parent_lit, parent_cost) = parent_node_idx
                    .map(|i| (self.nodes[i].literal, self.nodes[i].cost))
                    .unwrap_or_else(|| (true.into(), 0));

                // imply the lower one
                solver.add_clause(vec![!new_var, parent_lit]);

                self.nodes.push(CostTreeNode {
                    cost,
                    parent: parent_node_idx,
                    literal: new_var,
                });

                let cost_diff = cost - parent_cost;
                // println!("  - added cost duff {}", cost_diff);

                notify_vars(
                    // format!("{}_c{}_d{}", name, cost, cost_diff),
                    cost_diff,
                    new_var,
                );
            }
        }

        // Invert the tree pointers
        let mut pointers: HashMap<Option<usize>, Vec<usize>> = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            pointers.entry(node.parent).or_default().push(idx);
        }

        let emptyvec = Vec::new();
        let mut stack = pointers.get(&None).unwrap_or(&emptyvec).clone();
        let mut frontier = Vec::new();
        while let Some(node_idx) = stack.pop() {
            if self.nodes[node_idx].cost < cost {
                stack.extend(
                    pointers
                        .get(&Some(node_idx))
                        .unwrap_or(&emptyvec)
                        .iter()
                        .copied(),
                );
            } else {
                frontier.push(node_idx);
            }
        }

        // input lit implies at least this cost
        solver.add_clause(
            std::iter::once(!input).chain(frontier.into_iter().map(|n| self.nodes[n].literal)),
        );

        // let incoming_to_parent = self
        //     .nodes
        //     .iter()
        //     .filter(|n| n.parent == parent_node_idx)
        //     .collect::<Vec<_>>();

        // println!(
        //     "  - lower node {:?}",
        //     parent_node_idx.map(|i| &self.nodes[i])
        // );
        // println!("  - Higher nodes {:?}", incoming_to_parent);

        // if linear_mode {
        //     assert!(incoming_to_parent.len() <= 1);
        // }

        // if let Some(existing_node) = self.nodes.iter().find(|n| n.cost == cost) {
        //     let gte_incoming_to_parent = self
        //         .nodes
        //         .iter()
        //         .filter(|n| n.parent == existing_node.parent && n.cost >= cost);

        //     println!("  - Cost already in tree.");

        //     // input lit implies at least this cost
        //     solver.add_clause(
        //         std::iter::once(!input).chain(gte_incoming_to_parent.map(|n| n.literal)),
        //     );
        // } else {

        // }
    }
}

impl<L: satcoder::Lit + std::fmt::Debug> Default for CostTree<L> {
    fn default() -> Self {
        Self::new()
    }
}

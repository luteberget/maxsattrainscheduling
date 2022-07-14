# Train Re-scheduling using Dynamic Discretization Discovery

This repository contains a proof-of-concept Rust implementation of a train
re-scheduling algorithm based on the Dynamic Discretization Discovery
formulation using a SAT solver.  It also contains other solver implementations,
some unfinished. 

A paper describing the algorithm is currently under review.  The two solvers
compared in the paper are the MaxSAT DDD solver (in `src/solvers/maxsatddd.rs`)
and the Big-M solver (in `src/solvers/bigm.rs`).

## Problem instances

A set of 24 problem instances used for performance evaluation can be found in
the `instances_original` directory. These 24 instances are also modified to
become harder to solve. One modification adds more travel time in the tracks
(`original_addtracktime`), and another modification adds also more dwelling
time in stations (`original_addstationtime`).


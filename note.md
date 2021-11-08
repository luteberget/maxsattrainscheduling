---
title: An incremental MaxSAT formulation for on-line train scheduling
author: bjørnar
---

Re-scheduling trains after traffic disturbances is an important problem from
the operations research literature. 

There are several formulations of on-line train scheduling problem as
mixed-integer linear programming (MILP), including time-indexed\cite{ti}, big-M\cite{bigm},
path-and-cycle\cite{pnc}, and dynamic discretization discovery\cite{ddd}.  The constraints for the
problem are disjunctions of difference constraints over a continuous time
domain, though several of the MILP formulations are based on discretizing the
time domain to create a problem over purely binary variables.  However, since
the optimization objective is usually formulated as a continuous
piecewise-linear function, and it plays a crucial role in efficient train
scheduling, most algorithmic efforts have focused on classical optimization
techniques involving mixed-integer programming solvers (such as CPLEX and
Gurobi) or local search heuristics.

The recent dynamic discretization discovery (DDD) formulation altogether avoids
continuous variables for representing the arrival and departure times of train.
Instead, it solves an independent set problem, where
the nodes consist of a sequence of consecutive time intervals for each time variable,
and where edges are intervals that are incompatible under the difference constraints.
The algorithm lazily subdivides
the intervals and adds conflict edges between intervals until the difference constraints
are satisfied by the left-most point of each interval. Because the objective is
increasing in all time variables, selecting the earliest time from each
interval is a lower bound on the objective value, and thus the algorithm is exact.

We adapt this algorithm for a SAT solver, using the same idea of abstraction
refinement, though on unary-number-like representations of time-domain
variables.  The objective is discretized into a step-wise function (which has
also, independently, been suggested by domain experts), so that the resulting
problem becomes an incremental unweighted MaxSAT problem. We solve
this problem using  fully lazy abstraction refinement and conflict generation,
in combination with the RC2 MaxSAT algorithm.

# Problem definition

We simplify the on-line train scheduling problem using the following assumptions:

 * Trains may exclusively occupy single-track sections. 
 * There are no capacity constraints in stations, i.e. any train can use any
   track and/or platform, and there are no constraints on how many trains can
   be in the same station at the same time.
 * There are no safety margin on occupation times, i.e. a train can immediately leave 
   to a single-track section as soon as the previous train 
 * Trains take zero time to travel across a station.
 * Double tracks are treated as two oppositely directed single-track connections, used 
   exclusively for single-directed traffic.

We define the *simplified on-line train scheduling* input data as:

 * A set of resources $R$ and a binary relation defining the conflicting resources $C \subseteq R \times R$.
 * A set of trains $T$, each consisting of a sequence of visits $V_t^i$ for $i = 0, 1, \ldots$.
 * The visits $V_t^i$ are a three-tuple $V_t^i=(r_t^i,e_t^i,l_t^i)$, where 
   * $r_t^i \in R$ is a resource, 
   * $e_t^i \in \mathbb{R}$ is the earliest time the train can enter the resource, and 
   * $l_t^i \in \mathbb{R}$ is the minimum travel time the train needs to traverse the resource.


The constraints are:

 * Earliest time constraints: for each visit $V_t^i$,
$$ x_t^i \geq e_t^i $$
 * Travel time constraints: for two consecutive visits $V_t^i, V_t^{i+1}$, 
$$ x_t^i + l_t^i \leq x_t^{i+1} $$
 * Resource constraints: for any pair of visits $V_a^i, V_b^j$ that use conflicting resources $(r_a^i,r_b^j) \in C$,
$$ (r_a^{i+1} \geq r_b^j) \vee (r_b^{j+1} \geq r_a^i ) $$

The objective function is $$\sum_{t \in T} \sum_{V_t^i} \sigma(x_t^i - e_t^i),$$ where $x_t^i$ is the 
chosen time for train $t$ to start its visit $i$, and $\sigma(x)$ is:

 * $\sigma(x) = 3$, if $x > 360$ seconds
 * $\sigma(x) = 2$, if $x > 180$ seconds
 * $\sigma(x) = 1$, if $x > 0$
 * $\sigma(x) = 0$, otherwise

# Encoding

## Dynamically discretized number representation

We define a dynamically discretized number $y$ with a lower bound $lb(y)$ and upper bound $ub(y)$, as an increasing sequence of values and corresponding Boolean literals, initially containing:

$$ \left[ \left(lb(y), \top\right), \left(ub(y), \bot\right) \right] $$

We define the evaluation of the number $y$ in a propositional logic model $M$ as the value
corresponding with the last (right-most) element in the sequence which evaluates to true.
In the initial sequence, the evaluation will always be $lb(y)$. 


Whenever we need to add constraints involving expressions of the form $y \geq
c$, we check if $c$ is a value in the sequence. If it is, we can use the
corresponding Boolean literal to represent $y \geq c$.  If $c$ is not in the
sequence (and $lb(y) < c < ub(y)$), we can add it by creating a new Boolean variable $\lambda_y^c$
and inserting it in the sequence:

$$ \left[ \left(lb(y), \top\right), \left(c, \lambda_y^c \right), \left(ub(y), \bot\right) \right] $$

We also insert clauses to enforce consistency: each variable of the sequence
implies the previous.  Here, this is the non-clause $\lambda_y^c \Rightarrow
\top$, though in general for neighboring variables $\lambda_y^f$ and $\lambda_y^g$, we have:

$$ \lambda_y^g \Rightarrow \lambda_y^c, \quad \lambda_y^c \Rightarrow \lambda_y^f$$

Because variables are inserted into the sequence to maintain the values' ordering, we
always have $f < c < g$.  Note that if values are inserted in arbitrary order,
all the clauses are sufficient for consistency, and valid, though some become
redundant.

This encoding corresponds to the "unary" number representation \cite{björk}.

Note that the number may be either continuous or integral, in fact any totally
ordered domain would work. In practice, we use integers representing seconds
in the online train scheduling algorithm described below.

# Algorithm

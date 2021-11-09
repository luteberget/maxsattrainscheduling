---
title: An incremental MaxSAT formulation for on-line train scheduling
author: bjørnar
date: Note for Carlo 2021-11-09
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
  - \usepackage{lscape,longtable,booktabs}
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
Gurobi) or local search algorithms.

The recent dynamic discretization discovery (DDD) formulation  avoids altogether
continuous variables for representing the arrival and departure times of trains.
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
also, independently, been suggested by domain experts as a correct objective), so that the resulting
problem becomes an incremental unweighted MaxSAT problem. We solve
this problem using  fully lazy constraint generation,
in combination with the RC2 MaxSAT algorithm.

# Potential problems with this approach

 * The objective function does not change value at all when increasing
   the delay of a train above the maximum value. This allows
   the algorithm to effectively give up on optimizing a train that is sufficiently late, and delay it arbitrarily long. This seems to make the problem easier to solve in some cases, though the solution can be undesirable. For example, a train might be
   told to wait for hours in a station to keep the other trains on time.
 * There are some inefficient runs where visits with maximum
   cost can be involved in very many conflicts (two trains 
   can push each other forward in time indefinitely). This depends
   on the phase heuristic of the solver, and should be fixable.
 * The performance is not directly comparable to the MILP approaches
   because of the change in objective function. The objective function
   from the instance files are ignored.
 * The delays are minimized relative to the earliest possible arrival times,
   taking the delay into account, 
   but not taking into account which trains are more delayed at the current time.
   Also, any priority between trains (train types) is ignored.

# Problem definition

We simplify the on-line train scheduling problem using the following assumptions:

 * Trains may exclusively occupy single-track sections. 
 * There is no rerouting.
 * There are no capacity constraints in stations, i.e. any train can use any
   track and/or platform, and there are no constraints on how many trains can
   be in the same station at the same time.
 * There are no safety margin on occupation times, i.e. a train can immediately   enter  a single-track section as soon as the previous train has left, and multiple trains can enter a station simultaneously.
 * Trains occupy only one resource at a time (trains have effectively zero length).
 * Trains take zero time to travel across a station.
 * Double tracks are treated as two oppositely directed single-track connections, used 
   exclusively for single-directed traffic.

We define the *simplified on-line train scheduling* input data as:

 * A set of resources $R$ and a binary relation defining the conflicting resources $C \subseteq R \times R$.
 * A set of trains $t \in T$, each consisting of a sequence of visits $V_t^i$ for $i = 0, 1, \ldots$.
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
$$ (x_a^{i+1} \leq x_b^j) \vee (x_b^{j+1} \leq x_a^i ) $$

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
corresponding Boolean literal to represent $y \geq c$, and it's negation to represent $y < c$.  If $c$ is not in the
sequence (and $lb(y) < c < ub(y)$), we can add it by creating a new Boolean variable $\lambda_y^c$
and inserting it in the sequence:

$$ \left[ \left(lb(y), \top\right), \left(c, \lambda_y^c \right), \left(ub(y), \bot\right) \right] $$

We also insert clauses to enforce consistency: each variable of the sequence
implies the previous.  Here, this is the non-clause $\lambda_y^c \Rightarrow
\top$, though in general for neighboring variables $\lambda_y^f$ and $\lambda_y^g$, we have:

$$ \lambda_y^g \Rightarrow \lambda_y^c, \quad \lambda_y^c \Rightarrow \lambda_y^f$$

Because new values with corresponding fresh variables are inserted into the sequence at the correct place to preserve the ordering, we
always have $f < c < g$.  Note that when values can be incrementally
added in arbitrary order, all the clauses above are sufficient for consistency, and valid, though some clauses become redundant because the 
involved variables are no longer neighbors in the sequence.

This encoding is similar to the number representation known as the Unary
encoding\cite{björk} in that it represents lower and upper bounds, and uses 
one variable per possible value, but this version does not require all
values to be represented, and as such it is similar to the 
generalized totalizer encoding \cite{generalizedtotalizer}.

With this representation, we can create Boolean logic constraints involving 
$y < c$ and $y \geq c$ for any specific $c$.

Note that the number may be either continuous or integral, in fact any totally
ordered domain would work. In practice, we use integers representing seconds
in the online train scheduling algorithm described below.

## Discretized train scheduling constraints
Now, we implement each of the train scheduling constraints using dynamically
discretized number representations of each $x_t^i$ from the online train
scheduling problem.

 * **Earliest time constraints**: if we let $lb(x_t^i)=e_t^i$ (and $ub(x_t^i) =\infty$), then
   each constraint $x_t^i \geq e_t^i$ becomes $\top$, i.e. the constraint is implicit.

 * **Travel time constraint**: consider consecutive visits $x_t^i, x_t^{i+1}$.
   For every possible value $c$, we have the clause $$ x_t^i \geq c \Rightarrow x_t^{i+1} \geq c+l_t^i. $$
 * **Resource occupation constraint**: consider conflicting visits $x_a^i, x_b^j$. 
For every possible value $c_1$ and every possible value $c_2$, we have the clause
$$ \left( (x_a^{i+1} \geq c_1) \Rightarrow (x_b^j \geq c_1) \right) \vee \left( (x_b^{j+1} \geq c_2)  \Rightarrow (x_a^i \geq c_2) \right) $$

Although these are an infinte number of constraints, we can check a propositional
model M for violations of the original constraints, and add values $c$ to the
dynamically discretized numbers only as needed, and then also create the corresponding discretized constraints.

# Algorithm

\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Visits $V_t^i=(r_t^i, e_t^i, l_t^i)$ and objective function $\sigma(x)$.}
\Output{A value $x_t^i$ for each visit, fulfilling the train scheduling constraints and minimizing the objective function.}
\BlankLine
$\mathcal{S} \leftarrow $ new incremental MaxSAT solver instance\;
$\mathcal{O} \leftarrow \left\{ V_t^i:\, \left[ \left(l_t^i, \top\right), \left(\infty, \bot\right) \right]\ \right\}$\;
$\mathcal{C} \leftarrow \left\{ V_t^i:  \left[ (0,\top) \right] \right\}$\;
\While{true}{
    $M \leftarrow \mathcal{P}$\texttt{.solve}()\hspace{2em}(RC2 increases cost until SAT)\;
    $\left\{ x_t^i \right\} \leftarrow \left\{ \mathcal{O}[V_t^i]\text{\texttt{.evalDiscretizedNumber}}(M) \right\}$\;
    \ForEach{  $x_t^i + l_t^i > x_t^{i+1}$}{
      $\mathcal{O}[V_t^{i+1}]$\texttt{.newValue}$(x_t^i + l_t^i)$\;
      $\mathcal{P}$\texttt{.addClause}$( 
        \neg \mathcal{O}[V_t^{i}]\text{\texttt{.geq}}(x_t^i) \vee 
        \mathcal{O}[V_t^{i+1}]\text{\texttt{.geq}}(x_t^i+l_t^i)
        )$\;
    }
    \ForEach{ $ (x_a^{i+1} < x_b^j) \wedge (x_b^{j+1} < x_a^i )$}{
      $\mathcal{O}[V_a^{i}]$\texttt{.newValue}$(x_b^{j+1})$\;
      $\mathcal{O}[V_b^{j}]$\texttt{.newValue}$(x_a^{i+1})$\;
      $\mathcal{P}$\texttt{.addClause}$
        (
          \neg \mathcal{O}[V_b^{j+1}]\text{\texttt{.geq}}(x_b^{j+1}) \vee
               \mathcal{O}[V_a^i]\text{\texttt{.geq}}(x_b^{j+1})  \vee  
          \neg \mathcal{O}[V_a^{i+1}]\text{\texttt{.geq}}(x_a^{i+1}) \vee 
               \mathcal{O}[V_b^{j}]\text{\texttt{.geq}}(x_a^{i+1})  
        ) $\;
    }
    \If{no constraints were added}{
      \Return{$x_t^i$}
    }
    \ForEach{new value $c$ added to $V_t^i$}{
      \While{$\mathcal{C}[V_t^i]\text{\texttt{.maxValue}} < \sigma(c - e_t^i)$} {
      $\mathcal{C}[V_t^i]\text{\texttt{.extendUnary}}(\sigma(c - e_t^i))$\;
        $\mathcal{S}\text{\texttt{.addSoft}}(\neg \mathcal{C}[V_t^i]\text{\texttt{.geq}}(\mathcal{C}[V_t^i]\text{\texttt{.maxValue}}))$\;
      }
      $\mathcal{P}\text{\texttt{.addClause}}(
       \neg \mathcal{O}[V_a^{i+1}]\text{\texttt{.geq}}(x_a^{i+1}) \vee
            \mathcal{C}[V_t^i]\text{\texttt{.geq}}(\sigma(c - e_t^i))
      )$\;
    }
}
\caption{Incremental MaxSAT algorithm for the online train scheduling problem}
\end{algorithm} 

# Performance evaluation

The tables below show the performance of the 
algorithm on 20 instances supplied by Anna Livia Croella.
Each table uses a different objective function $\sigma^{a,b,c}$, where $\sigma(x)$ is:

 * $\sigma(x) = c$, if $x > 360$ seconds
 * $\sigma(x) = b$, if $x > 180$ seconds
 * $\sigma(x) = a$, if $x > 0$
 * $\sigma(x) = 0$, otherwise

Note that Instance 8 varies heavily in running time with different objective functions, while the other instance seem
to work the same.

The column headings are:

 * Insntc: the instance number.
 * Trains: the number of trains in the instance.
 * Resources: the number of single-track sections in the instance.
 * Avg.res.: the average number of single-track sectionss that trains need to traverse.
 * Confl.pairs: the total number of pairs of visits that share a resource.
 * Cost: the minimum of the objective function
 * $n_{SAT}$: the number of SAT problems solved that were satisfiable (resulting in further discretizing the domain).
 * $n_{UNSAT}$: the number of SAT problems solved that were unsatisfiable (resulting in further increasing the cost).
 * Travel confl.: the number of travel time constraints added.
 * Res. confl.: the number of resource conflict constraints adeded.
 * Vars: the number of Boolean variables in the final SAT problem.
 * Clauses: the number of clauses in the final SAT problem.
 * Solve time (ms): the total running time of the solver.


\input{perftable}
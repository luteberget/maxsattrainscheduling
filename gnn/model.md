# GNN Graph ideas

## Disjunctive graph

Wikipedia definition:
> If one task x must be performed earlier than a second task y, this constraint is represented by a directed edge from x to y. If, on the other hand, two tasks x and y can be performed in either order, but not simultaneously (perhaps because they both demand the use of the same equipment or other resource), this non-simultaneity constraint is represented by an undirected edge connecting x and y.

 * Node type "event".
 * Directed edge type "precedes" between events.
 * Undirected edge type "conflicts" between events.





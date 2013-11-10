Question 5: A set of states is reachable only if all of the states are achieved with the use of a single transition (0 or 1) because it "splits". There are no 3-way splits any set with 3 states in it is not reachable. Likewise there's no way to reach C alone because all the transitions that reach it also reach some other state as well.


Question 3:
proof reasons
- 1 for A or B
+ 2 for B
+? 3 for C
- 3 for A or B
- 4 for B
  5
+ 6 for A
- 7 for A or C
- 8 for B
- 9 for B
+ 10 for B


Question 5:
Brute force cracking method (discouraged by jeff ullman)
reachable states:
- {}
- {A}
- {B}
+ {C}
- {B, C}
- {A, C}
- {A, B}
+ {A, B, C}

Sorry. The empty set, or "dead state" is reachable from the start state {A} by inputs such as 11. Remember, when you perform the subset construction, keep track of all the reachable sets of NFA states. "Visit" each reachable set S in turn, and find the set of states reachable on 0 in the DFA from some member of S. That set of states is also reachable. Do the same for the set of states reachable from any member of S on input 1; that set is reachable as well.

Sorry. {A} is the start state, and therefore surely reachable. In particular, the empty string reaches {A}. Remember, when you perform the subset construction, keep track of all the reachable sets of NFA states. "Visit" each reachable set S in turn, and find the set of states reachable on 0 in the DFA from some member of S. That set of states is also reachable. Do the same for the set of states reachable from any member of S on input 1; that set is reachable as well.

Sorry. {B} is reachable from the start state {A} by input strings such as 1. Remember, when you perform the subset construction, keep track of all the reachable sets of NFA states. "Visit" each reachable set S in turn, and find the set of states reachable on 0 in the DFA from some member of S. That set of states is also reachable. Do the same for the set of states reachable from any member of S on input 1; that set is reachable as well.

Sorry. {A,C} is reachable from the start state {A} by input strings such as 10. Remember, when you perform the subset construction, keep track of all the reachable sets of NFA states. "Visit" each reachable set S in turn, and find the set of states reachable on 0 in the DFA from some member of S. That set of states is also reachable. Do the same for the set of states reachable from any member of S on input 1; that set is reachable as well.
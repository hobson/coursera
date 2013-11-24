In this assignment, you are expected to fill in programs that convert a regular expression to an ϵ-NFA. 

The parsing of the regular expression and the NFA data structure have been finished for you. What is left to fill in are the following functions in epsnfa.java: (Note: For Python version, the names of files, methods and variables are exactly the same as Java version)
//unite two ϵ-NFAs, with start state s1 and s2, final state t1 and t2, respectively 
//return an array of length 2, where the first element is the start state of the combined NFA. the second being the final state 
int[] union(int s1,int t1,int s2,int t2) 

//concatenation of two ϵ-NFAs, with start state s1 and s2, final state t1 and t2, respectively 
//return an array of length 2, where the first element is the start state of the combined NFA. the second being the final state 
int[] concat(int s1,int t1,int s2,int t2) 

//Closure of a ϵ-NFA, with start state s and final state t 
//return an array of length 2, where the first element is the start state of the closure ϵ-NFA. the second being the final state 
int[] clo(int s,int t) 

For definitions, you can refer to slides 14-16 of lecture 5 (Regular Expressions). You are expected to implement the functons that appear in diagram form in the slides. 
In each case, you should assign new states for the start state and final state of the resulting ϵ-NFA. You can do this via method: 
newstate=incCapacity(); 
To add edges, you can use methods: 
addEdge(a,0,b); //add an edge with symbol 0 from a to b 
or addEdge(a,1,b); //add an edge with symbol 1 from a to b 
or addEdge(a,epssymbol,b); //add an edge with symbol ϵ from a to b 
Basically, the whole program, including what you write, will read regular expressions and transform them one by one into an ϵ-NFA. Then the program will check whether the ϵ-NFA you construct is correct. 

sample input (sampleRE.in): 
1.0.1 
0+1 
0+1.0.1 
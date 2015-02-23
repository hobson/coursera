Hobson Lane
presentation on 
    AI Planning for Data Mining 
        Using A-Star Search
submission for the
    Creative Challenge assignment
    for the Performance Level 
    of the AI Planning Class at Coursera
    taught by Wickler and Tate 
    and numerous other contributors
First I'd like to review the AStar algorithm
    and it's application to the Eight-Puzzle toy problem
Here is my adaptation of the AIMA python package 
    by Norvig and Russel
    I've added support for non-hashable states within the Node class and some simple functions for state
    transitions in the Tile Puzzle problems
    I've also simplified the A-Star algorithm implementation
    so that it matches precisely the implementation provided by Wickler and Tate
    You can instantiate a Tile shuffling problem with any number of tiles (for square boards) by simply specifying the initial state as a sequence or array of integers from zero to N, N is a perfect square 
This code is on [github](http://github.com/hobson/coursera/) within the aiplan packaged.
    I've also packaged my modified AIMA package with a setup file so it is installable from the python cheese shop with `pip install aima`.
Finally here is an applicaiton in the real world for my employer.
    We're using Astar search to find ways to save money by reducing product returns at my employer.
    The goal in this case is to find a high correlation between one field in a database and the field that records our returns over time as they flow into our service center.

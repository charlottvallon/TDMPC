# Task Decomposition for Iterative Learning Model Predictive Control (TDMPC)

TDMPC is a data-driven method for finding a feasible trajectory to smartly initialize a Learning Model Predictive Controller for a new task, using data from previous tasks. Learning Model Predictive Control (LMPC) is a data-driven control framework developed by Ugo Rolia in the MPC lab at UC Berkeley. In this example, we implement TDMPC for an autonomous racing task. First, an LMPC controller drives several laps of a race track, learning from experience to improve the lap time at each iteration. After the LMPC has converged, a new race track is constructed, consisting of the same road segments as the first track. Then, using stored learning data from the first track, TDMPC is used to design a feasible trajectory and MPC-based policy for the new, unseen track. 

## Workflow

1) A PID path-following controller is used to drive the vehicle around the original track.
2) The trajectory data from step 1 is used to estimate an LTI model, used to design an MPC for path-following around the track.
3) The data from step 2 is used to estimate an LTV model, used to design an MPC for path-following around the track.
4) The data from steps 1-3 are used to build a "safe set" and a terminal cost, which are used to initialize the LMPC. The LMPC uses a LTV model identified from data.
5) A new track is created, using road segments from the original track. TDMPC is used to build a safe set and terminal cost for the new track, along with an initial feasible trajectory.
6) A new LMPC can be initialized for the new track using the TDMPC-derived safe set and terminal cost. 

## Results
We evaluate two different LMPC controllers on the new track: 
   - an LMPC controller initialized using Task Decomposition (TDMPC), as described above
   - a basic LMPC controller initialized using a PID path-following controller on the new track
   
Our results demonstrate that the TDMPC allows an LMPC to converge to an optimal minimum-time trajectory faster than finding an initial feasible trajectory using simple initialization methods such as the PID path-following controller.

<p align="center">
<img src="https://github.com/urosolia/RacingLMPC/blob/master/src/ClosedLoop.gif" width="500" />
</p>

## References

This code is based on the following publications:

* Charlott Vallon and Francesco Borrelli. "Task Decomposition for Iterative Learning Model Predictive Control." In IEEE 2020 American Control Conference (ACC). [PDF](http://128.84.4.27/abs/1903.07003)
* Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)

## Questions

Please direct any questions you have about this code to charlottvallonATberkeley.edu

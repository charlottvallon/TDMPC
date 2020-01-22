# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
# [2] Charlott Vallon and Francesco Borrelli. "Task Decomposition for Iterative Learning Model Predictive Control."
#     In 2020 IEEE American Control Conference (ACC)
#
# Attibution Information: Code developed by Charlott Vallon and Ugo Rosolia
# (for clarifications and suggestions please write to charlottvallon@berkeley.edu, ugo.rosolia@berkeley.edu).
#
# Code description: Simulation of the Task Decomposition for Iterative Learning Model Predictive Controller (TDMPC). 
# The main file runs:
# 1) A PID path following controller
# 2) A Model Predictive Controller (MPC) which uses a LTI model identified from the data collected with the PID in 1)
# 3) A MPC which uses a LTV model identified from the date collected in 1)
# 4) A LMPC for racing where the safe set and value function approximation are build using the data from 1), 2) and 3)
# 5) Two versions of the LMPC controller on an unseen track, consisting of the same track segments as parts 1)-4):
#   - an LMPC controller initialized using Task Decomposition (TDMPC)
#   - a basic LMPC controller initialized using a PID path following controller
# 6) An option for Multi-Shuffling, which designs LMPC safe sets using TDMPC from several different tasks (rather than
#    just one previously seen subtask sequence). This is associated with more drastic cost improvement over basic LMPC.
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('fnc')
from SysModel import Simulator, PID
from Classes import ClosedLoopData, LMPCprediction
from PathFollowingLTVMPC import PathFollowingLTV_MPC
from PathFollowingLTIMPC import PathFollowingLTI_MPC
from Track import Map
from LMPC import ControllerLMPC, PWAControllerLMPC
from Utilities import Regression
from plot import plotTrajectory, plotClosedLoopLMPC, plotMap, plotSafeSet
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ======================================================================================================================
# ============================ Choose which controller to run to set up problem ========================================
# ======================================================================================================================
RunPID     = 0; plotFlag       = 1
RunMPC     = 0; plotFlagMPC    = 1
RunMPC_tv  = 0; plotFlagMPC_tv = 1
RunLMPC    = 0; plotFlagLMPC   = 1; animation_xyFlag = 1; animation_stateFlag = 0
RunShuffle = 1;
runPWAFlag = 0; # uncomment importing pwa_cluster in LMPC.py
testCoordChangeFlag = 0;
plotOneStepPredictionErrors = 1;

# ======================================================================================================================
# ============================ Initialize parameters for path following ================================================
# ======================================================================================================================
dt         = 1.0/10.0        # Controller discretization time
Time       = 100             # Simulation time for path following PID
TimeMPC    = 100             # Simulation time for path following MPC
TimeMPC_tv = 100             # Simulation time for path following LTV-MPC
vt         = 0.8             # Reference velocity for path following controllers
v0         = 0.5             # Initial velocity at lap 0
N          = 12            # Horizon length
n = 6;   d = 2               # State and Input dimension

# Path Following tuning
Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
R = np.diag([1.0, 10.0])                  # delta, a

map = Map(0.8)                            # Initialize the map (PointAndTangent); argument is track width
simulator = Simulator(map)                # Initialize the Simulator

# ======================================================================================================================
# ==================================== Initialize parameters for LMPC ==================================================
# ======================================================================================================================
TimeLMPC   = 400              # Simulation time
Laps       = 5+2              # Total LMPC laps
ShuffledLaps = 2+10

# Safe Set Parameter
LMPC_Solver = "CVX"           # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
numSS_it = 2                  # Number of trajectories used at each iteration to build the safe set
numSS_Points = 32 + N         # Number of points to select from each trajectory to build the safe set
shift = 0                     # Given the closed point, x_t^j, to the x(t) select the SS points from x_{t+shift}^j

# Tuning Parameters
Qslack  = 5*np.diag([10, 1, 1, 1, 10, 1])          # Cost on the slack variable for the terminal constraint
Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
R_LMPC  =  1 * np.diag([1.0, 1.0])                      # Input cost u = [delta, a]
dR_LMPC =  5 * np.array([1.0, 1.0])                     # Input rate cost u

# Initialize LMPC simulator
LMPCSimulator = Simulator(map, 1, 1) #flags indicate one lap, and using the LMPC controller

# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
print("Starting PID")
if RunPID == 1:
    ClosedLoopDataPID = ClosedLoopData(dt, Time , v0) #form matrices for experiment data
    PIDController = PID(vt) #sets the reference velocity and some timers?
    simulator.Sim(ClosedLoopDataPID, PIDController) #simulates the PID controller for Time timesteps

    file_data = open('data/ClosedLoopDataPID.obj', 'wb')
    pickle.dump(ClosedLoopDataPID, file_data)
    file_data.close()
else:
    file_data = open('data/ClosedLoopDataPID.obj', 'rb')
    ClosedLoopDataPID = pickle.load(file_data)
    file_data.close()
print("===== PID terminated")
if plotFlag == 1:
    plotTrajectory(map, ClosedLoopDataPID.x, ClosedLoopDataPID.x_glob, ClosedLoopDataPID.u)
    plt.show()
    
# ======================================================================================================================
# ======================================  LINEAR REGRESSION ============================================================
# ======================================================================================================================
raw_input("Finished PID - Start TI-MPC?")
print("Starting TI-MPC")
lamb = 0.0000001
#fit linear dynamics to the closed loop data: x2 = A*x1 + b*u1; lamb is weight on frob norm of W
A, B, Error = Regression(ClosedLoopDataPID.x, ClosedLoopDataPID.u, lamb)

if RunMPC == 1:
    ClosedLoopDataLTI_MPC = ClosedLoopData(dt, TimeMPC, v0) #form (empty) matrices for experiment data
    Controller_PathFollowingLTI_MPC = PathFollowingLTI_MPC(A, B, Q, R, N, vt)
    simulator.Sim(ClosedLoopDataLTI_MPC, Controller_PathFollowingLTI_MPC)

    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'wb')
    file_data = open('data/ClosedLoopDataLTI_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTI_MPC, file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'rb')
    file_data = open('data/ClosedLoopDataLTI_MPC.obj', 'rb')
    ClosedLoopDataLTI_MPC = pickle.load(file_data)
    file_data.close()
print("===== TI-MPC terminated")
if plotFlagMPC == 1:
    plotTrajectory(map, ClosedLoopDataLTI_MPC.x, ClosedLoopDataLTI_MPC.x_glob, ClosedLoopDataLTI_MPC.u)
    plt.show()
# ======================================================================================================================
# ===================================  LOCAL LINEAR REGRESSION =========================================================
# ======================================================================================================================
raw_input("Finished TI-MPC - Start TV-MPC?")
print("Starting TV-MPC")
if RunMPC_tv == 1:
    ClosedLoopDataLTV_MPC = ClosedLoopData(dt, TimeMPC_tv, v0)
    Controller_PathFollowingLTV_MPC = PathFollowingLTV_MPC(Q, R, N, vt, n, d, ClosedLoopDataPID.x, ClosedLoopDataPID.u, dt, map)
    simulator.Sim(ClosedLoopDataLTV_MPC, Controller_PathFollowingLTV_MPC)

    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'wb')
    file_data = open('data/ClosedLoopDataLTV_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTV_MPC, file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'rb')
    file_data = open('data/ClosedLoopDataLTV_MPC.obj', 'rb')
    ClosedLoopDataLTV_MPC = pickle.load(file_data)
    file_data.close()
print("===== TV-MPC terminated")
if plotFlagMPC_tv == 1:
    plotTrajectory(map, ClosedLoopDataLTV_MPC.x, ClosedLoopDataLTV_MPC.x_glob, ClosedLoopDataLTV_MPC.u)
    plt.show()

# ======================================================================================================================
# ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
# ======================================================================================================================
raw_input("Finished TV-MPC - Start LMPC?")
print("Starting LMPC")
ClosedLoopLMPC = ClosedLoopData(dt, TimeLMPC, v0)
LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps) #to store open-loop prediction and safe sets
LMPCSimulator = Simulator(map, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON

if runPWAFlag == 1:
    LMPController = PWAControllerLMPC(10, numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
else:
    LMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
    onlyLMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, 10*Laps, TimeLMPC, LMPC_Solver)

# add previously completed trajectories to Safe Set: 
LMPController.addTrajectory(ClosedLoopDataPID)
LMPController.addTrajectory(ClosedLoopDataLTV_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopLMPC.x[0,:]
x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

if RunLMPC == 1:
    for it in range(2, Laps):

        ClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        LMPCSimulator.Sim(ClosedLoopLMPC, LMPController, LMPCOpenLoopData) #this runs one lap at a time due to initialization!
        LMPController.addTrajectory(ClosedLoopLMPC)
        onlyLMPController.addTrajectory(ClosedLoopLMPC)

        if LMPController.feasible == 0:
            break
        else:
            # Reset Initial Conditions
            x0[0,:]      = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
            x0_glob[0,:] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]

    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'wb')
    file_data = open('data/LMPController.obj', 'wb')
    pickle.dump(ClosedLoopLMPC, file_data)
    pickle.dump(LMPController, file_data)
    pickle.dump(LMPCOpenLoopData, file_data)
    file_data.close()
    
    file_data = open('data/onlyLMPController.obj','wb')
    pickle.dump(onlyLMPController,file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'rb')
    file_data = open('data/LMPController.obj', 'rb')
    ClosedLoopLMPC = pickle.load(file_data)
    LMPController  = pickle.load(file_data)
    LMPCOpenLoopData  = pickle.load(file_data)
    file_data.close()
    
    file_data = open('data/onlyLMPController.obj','rb')
    onlyLMPController = pickle.load(file_data)

print("===== LMPC terminated")

if plotFlagLMPC == 1:
    plotClosedLoopLMPC(LMPController, map)
    plt.show()

# plot the safe set along the map 
plotSafeSet(onlyLMPController.SS,map)
raw_input("LMPC on original track is done.")


# ======================================================================================================================
# ======================================================= TDMPC ========================================================
# ======================================================================================================================
# We will now compare performance of the LMPC controller on the shuffled track. In particular, we consider two cases:
#   1. Performance of the LMPC controller initialized with the PID Safe Set
#   2. Performance of the LMPC controller initialized with TDMPC 
# Performance will be evaluated on iterations required to traverse the course

Shuffling_Iterations = 0

# if > 1, MultiShuffling is activated. 
DesiredShuffles = 1 
Cost_Improvement = np.zeros((DesiredShuffles,1))

if RunShuffle == 1:
    while Shuffling_Iterations < DesiredShuffles:
        infFlag = False # start by assuming feasibility
        shuffledWell = False
        onlyLMPController.processQfun()
        # split safe set into modes
        onlyLMPController.splitTheSS(map)          
        # relativize safe set (set intial s --> 0)
        onlyLMPController.relTheSplitSS(map)          
       
        while not shuffledWell: 
            shuffledMap = map.shuffle()    
            plotMap(shuffledMap)
            onlyLMPController.makeShuffledSS(shuffledMap) # turn relative safe set into absolute coordinates again (in modes)
            plotSafeSet(onlyLMPController.shuffledSS, shuffledMap)
            onlyLMPController.reachabilityAnalysis(A,B,Qslack,N)          
            shuffledWell = bool(input("Shuffled correctly?"))   
        
        plotSafeSet(onlyLMPController.reachableSS,shuffledMap)
        onlyLMPController.reorganizeReachableSafeSet()
        raw_input("Reachability analysis on new track is done.")
        
        # ========================================= 0. Set up for simulation on new track ======================================
        simulatorPID = Simulator(shuffledMap) 
        simulatorPID.laps = 1
        
        # ========================================= 0a. Run the PID controller on new track =====================================
        ClosedLoopDataShuffledPID = ClosedLoopData(dt, 0.5*Time , v0) #gets reset
        ShuffledPIDController = PID(vt) #sets the reference velocity and some timers?
        simulatorPID.Sim(ClosedLoopDataShuffledPID, ShuffledPIDController) #simulates the PID controller for Time timesteps
        file_data = open('data/ClosedLoopDataShuffledPID.obj', 'wb')
        pickle.dump(ClosedLoopDataShuffledPID, file_data)
        file_data.close()
        
        plotTrajectory(shuffledMap, ClosedLoopDataShuffledPID.x, ClosedLoopDataShuffledPID.x_glob, ClosedLoopDataShuffledPID.u)
        plt.show()
        
        # ================================= 1. Create and run TDMPC-initialized LMPC controller ================================= 
        raw_input("Going to start running Shuffled LMPC")
        # This controller will start with a TDMPC-initialized safe set
         
        ClosedLoopTDMPC = ClosedLoopData(dt, TimeLMPC, v0)
        TDMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
        TDMPCSimulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
        TDMPCController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)
        
        rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
        TDMPCController.addReachableSet(rSS,ruSS,rQfun,shuffledMap)
        TDMPCController.addTrajectory(ClosedLoopDataShuffledPID)
               
        x0           = np.zeros((1,n))
        x0_glob      = np.zeros((1,n))
        x0[0,:]      = ClosedLoopTDMPC.x[0,:]
        x0_glob[0,:] = ClosedLoopTDMPC.x_glob[0,:]
        
        for it in range(2, ShuffledLaps):
            ClosedLoopTDMPC.updateInitialConditions(x0, x0_glob)
            TDMPCSimulator.Sim(ClosedLoopTDMPC, TDMPCController, TDMPCOpenLoopData) #this runs one lap at a time due to initialization!
            TDMPCController.addTrajectory(ClosedLoopTDMPC)
              
            if TDMPCController.feasible == 0:
                infFlag = True
                break
            else:
                # add successful trajectory
                onlyLMPController.addTrajectoryToSS(onlyLMPController.shuffledSS, onlyLMPController.shuffleduSS, onlyLMPController.shuffledQfun, ClosedLoopTDMPC)
                # Reset Initial Conditions
                x0[0,:]      = ClosedLoopTDMPC.x[0,:]
                x0_glob[0,:] = ClosedLoopTDMPC.x_glob[0,:]       
        if infFlag:
            continue
        
        plt.figure()
        plotClosedLoopLMPC(TDMPCController,shuffledMap)
        plt.title('MBTTL Initialized')
        plt.show()
        
        file_data = open('data/TDMPCController.obj','wb')
        pickle.dump(TDMPCController,file_data)
        file_data.close()
    
        # =============================== 2. Create and run PID-initialized LMPC controller ==================================== 
        raw_input("Going to start running PID-initialized LMPC Controller")
        # This controller will start with a PID-initialized safe set
        
        ClosedLoopBasicLMPC = ClosedLoopData(dt, TimeLMPC, v0)
        BasicLMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
        BasicLMPCSimulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
        BasicLMPCController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)
        
        BasicLMPCController.addTrajectory(ClosedLoopDataShuffledPID)
        BasicLMPCController.addTrajectory(ClosedLoopDataShuffledPID)
        
        x0           = np.zeros((1,n))
        x0_glob      = np.zeros((1,n))
        x0[0,:]      = ClosedLoopBasicLMPC.x[0,:]
        x0_glob[0,:] = ClosedLoopBasicLMPC.x_glob[0,:]
        
        for it in range(2, ShuffledLaps):
            ClosedLoopBasicLMPC.updateInitialConditions(x0, x0_glob)
            BasicLMPCSimulator.Sim(ClosedLoopBasicLMPC, BasicLMPCController, BasicLMPCOpenLoopData) #this runs one lap at a time due to initialization!
            BasicLMPCController.addTrajectory(ClosedLoopBasicLMPC)
        
            if BasicLMPCController.feasible == 0:
                infFlag = True
                break
            else:
                x0[0,:]      = ClosedLoopBasicLMPC.x[0,:]
                x0_glob[0,:] = ClosedLoopBasicLMPC.x_glob[0,:]
                
        if infFlag:
            continue
        
        # laps over time
        plt.figure()
        plotClosedLoopLMPC(BasicLMPCController,shuffledMap)
        plt.title('PID Initialized')
        plt.show()
               
        # ================================ 3. Compare performance between controllers ========================================= 
        # lap time vs. lap
        plt.figure()
        plt.plot(range(1,BasicLMPCController.it-1), BasicLMPCController.Qfun[0,2:]*dt,label='PID-initialized')
        plt.plot(range(1,TDMPCController.it-1), TDMPCController.Qfun[0,2:]*dt,label='TDMPC-initialized')
        plt.legend()
        plt.title('Lap Time vs. Lap')
        plt.show()
        
        # save for later
        file_data = open('data/ShuffledData.obj','wb')
        pickle.dump(BasicLMPCController,file_data)
        pickle.dump(TDMPCController,file_data)
        pickle.dump(shuffledMap,file_data)
        file_data.close()
    
        print 'Finished'
        print('Cost_Improvement: ', Cost_Improvement)
        
        # update this in order to do another shuffling round.
        map  = shuffledMap
        Cost_Improvement[Shuffling_Iterations] = 100*np.sum(TDMPCController.Qfun[0,2:]*dt) / np.sum(BasicLMPCController.Qfun[0,2:]*dt)
        Shuffling_Iterations += 1
               
else:
    # load the map and controllers
    file_data = open('data/ShuffledData.obj', 'rb')
    BasicLMPCController = pickle.load(file_data)
    TDMPCController = pickle.load(file_data)
    shuffledMap = pickle.load(file_data)
    file_data.close()
    
    # plot the time improvement
    plt.figure()
    plt.plot(range(1,BasicLMPCController.it-1), BasicLMPCController.Qfun[0,2:]*dt,label='PID-initialized')
    plt.plot(range(1,TDMPCController.it-1), TDMPCController.Qfun[0,2:]*dt,label='TDMPC-initialized')
    plt.legend()
    plt.title('Lap Time vs. Lap')
    plt.show()
    
    
# End

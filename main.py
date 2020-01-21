# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
#
# [2] Ugo Rosolia, Ashwin Carvalho, and Francesco Borrelli. "Autonomous racing using learning model predictive control."
#     In 2017 IEEE American Control Conference (ACC)
#
# [3] Maximilian Brunner, Ugo Rosolia, Jon Gonzales and Francesco Borrelli "Repetitive learning model predictive
#     control: An autonomous racing example" In 2017 IEEE Conference on Decision and Control (CDC)
#
# [4] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally
#     Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017).
#
# Attibution Information: Code developed by Ugo Rosolia
# (for clarifications and suggestions please write to ugo.rosolia@berkeley.edu).
#
# Code description: Simulation of the Learning Model Predictive Controller (LMPC). The main file runs:
# 1) A PID path following controller
# 2) A Model Predictive Controller (MPC) which uses a LTI model identified from the data collected with the PID in 1)
# 3) A MPC which uses a LTV model identified from the date collected in 1)
# 4) A LMPC for racing where the safe set and value function approximation are build using the data from 1), 2) and 3)
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('fnc')
import os
from os.path import expanduser
from SysModel import Simulator, PID
from Classes import ClosedLoopData, LMPCprediction
from PathFollowingLTVMPC import PathFollowingLTV_MPC
from PathFollowingLTIMPC import PathFollowingLTI_MPC
from Track import Map, unityTestChangeOfCoordinates
from LMPC import ControllerLMPC, PWAControllerLMPC
from Utilities import Regression
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation, plotMap, plotSafeSet, plotTrajectoriesComparison, plotQprogression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pdb
import pickle
import random

# ======================================================================================================================
# ============================ Choose which controller to run to set up problem ========================================
# ======================================================================================================================
RunPID     = 0; plotFlag       = 1
RunMPC     = 0; plotFlagMPC    = 1
RunMPC_tv  = 0; plotFlagMPC_tv = 1
RunLMPC    = 1; plotFlagLMPC   = 1; animation_xyFlag = 1; animation_stateFlag = 0
runPWAFlag = 0; # uncomment importing pwa_cluster in LMPC.py
testCoordChangeFlag = 0;
plotOneStepPredictionErrors = 1;

DesiredShuffles = 1

random.seed()
directory_index = random.randint(1, 1000)
home = expanduser('~')
dl_path = home + '/Desktop/RacingLMPC/'+str(directory_index)
os.makedirs(dl_path)
# ======================================================================================================================
# ============================ Initialize parameters for path following ================================================
# ======================================================================================================================
dt         = 1.0/10.0        # Controller discretization time
Time       = 100             # Simulation time for path following PID
TimeMPC    = 100             # Simulation time for path following MPC
TimeMPC_tv = 100             # Simulation time for path following LTV-MPC
vt         = 0.8             # Reference velocity for path following controllers
v0         = 0.5             # Initial velocity at lap 0
N          = 12              # Horizon length
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
    onlyLMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, (Laps + (DesiredShuffles+1)*3*7), TimeLMPC, LMPC_Solver)

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
            #x0[0,:]      = ClosedLoopLMPC.x[0,:]
            #x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

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

Shuffling_Iterations = 0

# <codecell>
# ======================================================================================================================
# ========================================= TRACK/SAFE SET RESHUFFLING =================================================
# ======================================================================================================================


# run these simulations for longer to show convergence behavior to optimal trajectory
Laps = 10
ShuffledLaps = 10

Cost_Improvement = np.zeros((DesiredShuffles,1))

while Shuffling_Iterations < DesiredShuffles:

    onlyLMPController.processQfun()
    #   SS (u, SS, Q)
    
    # split safe set into modes
    onlyLMPController.splitTheSS(map)
    #   SS (u, SS, Q)
    #   splitSS (u,SS,Q)
        
    # relativize safe set (set intial s --> 0)
    onlyLMPController.relTheSplitSS(map)
    #   SS (u, SS, Q)
    #   splitSS (u,SS,Q)
    #   relSplitSS (SS)
       
    shuffledWell = False
    while not shuffledWell: 
        # shuffle safe set according to new track
        shuffledMap = map.shuffle()    
    
        onlyLMPController.makeShuffledSS(shuffledMap) #uses as input self.SS, uSS, Qfun
        # shuffledSplitSS (SS, uSS, Q)
        # shuffledSS (SS, uSS, Q)
        
        onlyLMPController.reachabilityAnalysis(A,B,Qslack,N)
        # reachableSS (SS, uSS, Qfun)
        # reachableSplitSS (SS, uSS, Qfun)
        
        shuffledWell = bool(input("Shuffled correctly?"))   
    
    #plotSafeSet(onlyLMPController.reachableSS,shuffledMap)
    #onlyLMPController.reorganizeReachableSafeSet() for some reason this fucks everything up
    rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory() #out comes one best trajectory
    
    # ========================================= 0. Set up for simulation on new track ======================================
    simulatorPID = Simulator(shuffledMap) 
    simulatorPID.laps = 1
    
    # ========================================= 0a. Run the PID controller on new track =====================================
    ClosedLoopDataShuffledPID = ClosedLoopData(dt, 0.5*Time , v0) #form matrices for experiment data
    ShuffledPIDController = PID(vt) #sets the reference velocity and some timers?
    simulatorPID.Sim(ClosedLoopDataShuffledPID, ShuffledPIDController) #simulates the PID controller for Time timesteps
    file_data = open('data/ClosedLoopDataShuffledPID.obj', 'wb')
    pickle.dump(ClosedLoopDataShuffledPID, file_data)
    file_data.close()
    
    plotTrajectory(shuffledMap, ClosedLoopDataShuffledPID.x, ClosedLoopDataShuffledPID.x_glob, ClosedLoopDataShuffledPID.u)
    plt.show()
    
    # ========================================= 2 Create and run LMPC controller 2 ========================================= 
    raw_input("Going to start running Shuffled LMPC2")
    # This controller will start with a safe set consisting of RSS and PID laps
     
    
    ClosedLoopShuffledLMPC21 = ClosedLoopData(dt, TimeLMPC, v0)
    ShuffledLMPC21OpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
    ShuffledLMPC21Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
    ShuffledLMPC21Controller = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)
    
    rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
    ShuffledLMPC21Controller.addReachableSet(rSS,ruSS,rQfun,shuffledMap) #################### FIX THIS FIX THIS
    ShuffledLMPC21Controller.addTrajectory(ClosedLoopDataShuffledPID)
    
    
    x0           = np.zeros((1,n))
    x0_glob      = np.zeros((1,n))
    x0[0,:]      = ClosedLoopShuffledLMPC21.x[0,:]
    x0_glob[0,:] = ClosedLoopShuffledLMPC21.x_glob[0,:]
    
    for it in range(2, ShuffledLaps):
        #ShuffledLMPC2Controller.numSS_it = ShuffledLMPC2Controller.it
        ClosedLoopShuffledLMPC21.updateInitialConditions(x0, x0_glob)
        ShuffledLMPC21Simulator.Sim(ClosedLoopShuffledLMPC21, ShuffledLMPC21Controller, ShuffledLMPC21OpenLoopData) #this runs one lap at a time due to initialization!
        ShuffledLMPC21Controller.addTrajectory(ClosedLoopShuffledLMPC21)
        onlyLMPController.addTrajectoryToSS(onlyLMPController.shuffledSS, onlyLMPController.shuffleduSS, onlyLMPController.shuffledQfun, ClosedLoopShuffledLMPC21)
    
        if ShuffledLMPC21Controller.feasible == 0:
            break
        else:
            # Reset Initial Conditions
            x0[0,:]      = ClosedLoopShuffledLMPC21.x[0,:]
            x0_glob[0,:] = ClosedLoopShuffledLMPC21.x_glob[0,:]
    
    
    plotClosedLoopLMPC(ShuffledLMPC21Controller,shuffledMap)
    plt.show()
            
            
            
            
            
    raw_input("Compare against PID next")
        
    # ========================================= Check Performance against PID-Only =====================================
    #### THIS IS WHERE TO CHECK PERFORMANCE AGAINST PID-ONLY 
    
    # run the PID-initialized LMPC (on same N as training)
    ClosedLoopShuffledLMPC_PID = ClosedLoopData(dt, TimeLMPC, v0)
    ShuffledLMPC_PIDOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps) #to store open-loop prediction and safe sets
    ShuffledLMPC_PIDSimulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
    ShuffledLMPC_PIDController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, Laps, TimeLMPC, LMPC_Solver)
    
    ShuffledLMPC_PIDController.addTrajectory(ClosedLoopDataShuffledPID)
    ShuffledLMPC_PIDController.addTrajectory(ClosedLoopDataShuffledPID)
    
    x0           = np.zeros((1,n))
    x0_glob      = np.zeros((1,n))
    x0[0,:]      = ClosedLoopShuffledLMPC_PID.x[0,:]
    x0_glob[0,:] = ClosedLoopShuffledLMPC_PID.x_glob[0,:]
    
    for it in range(2, Laps):
        #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
        ClosedLoopShuffledLMPC_PID.updateInitialConditions(x0, x0_glob)
        ShuffledLMPC_PIDSimulator.Sim(ClosedLoopShuffledLMPC_PID, ShuffledLMPC_PIDController, ShuffledLMPC_PIDOpenLoopData) #this runs one lap at a time due to initialization!
        ShuffledLMPC_PIDController.addTrajectory(ClosedLoopShuffledLMPC_PID)
    
        if ShuffledLMPC_PIDController.feasible == 0:
            break
        else:
            x0[0,:]      = ClosedLoopShuffledLMPC_PID.x[0,:]
            x0_glob[0,:] = ClosedLoopShuffledLMPC_PID.x_glob[0,:]
    
    
    # plot the results
    plotQprogression(ShuffledLMPCController, ShuffledLMPC_PIDController,dt,shuffledMap,Shuffling_Iterations,dl_path)
    Cost_Improvement[Shuffling_Iterations] = 100*np.sum(ShuffledLMPCController.Qfun[0,2:]*dt) / np.sum(ShuffledLMPC_PIDController.Qfun[0,2:]*dt)
    
    # ========================================= Reset variables for repeat =====================================
    #### RESET EVERYTHING
    map = shuffledMap
    
    # reset the controller SS to the shuffled SS
    onlyLMPController.SS = onlyLMPController.shuffledSS
    onlyLMPController.uSS = onlyLMPController.shuffleduSS
    onlyLMPController.Qfun = onlyLMPController.shuffledQfun
    
    
    Shuffling_Iterations += 1    
    raw_input("Start another shuffling iteration?")

# Plot the overall comparisons
plt.plot(range(1,DesiredShuffles+1),Cost_Improvement[:,0])
plt.title('Cost Improvement with Reachable Safe Set Initialization')
plt.xlabel('Track Recombinations')
plt.ylabel('100 * (Cost of RSS Initialization / Cost of PID Initialization)')
plt.savefig(dl_path+'/Cost_Improvement.eps')
plt.show()

# <codecell> PLOTTING FOR CDC PAPER
#file_data = open('data/GoodShuffledData.obj', 'rb')
#SS = pickle.load(file_data)
#file_data.close()
#
##plotSafeSet(GoodShuffledData,shuffledMap)
#
#plt.figure()
#counter = 0
#SSPoints = np.zeros([SS.shape[0],2])
#for lap in range(0,SS.shape[2]):
#    for state in range(0,SS.shape[0]):
#        if SS[state,0,lap]<1000:
#            #print(counter)
#            try:
#                SSPoints[counter,:] = shuffledMap.getGlobalPosition(SS[state,4,lap], SS[state,5,lap])
#                counter += 1
#            except TypeError:
#                print("something about the nonetype again")
#                
#plt.plot(SSPoints[:,0],SSPoints[:,1],'.r')
#
#Points = int(np.floor(10 * (shuffledMap.PointAndTangent[-1, 3] + shuffledMap.PointAndTangent[-1, 4])))
#Points1 = np.zeros((Points, 2))
#Points2 = np.zeros((Points, 2))
#Points0 = np.zeros((Points, 2))
#for i in range(0, int(Points)):
#    Points1[i, :] = shuffledMap.getGlobalPosition(i * 0.1, shuffledMap.width)
#    Points2[i, :] = shuffledMap.getGlobalPosition(i * 0.1, -shuffledMap.width)
#    Points0[i, :] = shuffledMap.getGlobalPosition(i * 0.1, 0)
#
#plt.plot(shuffledMap.PointAndTangent[:, 0], shuffledMap.PointAndTangent[:, 1], 'o')
#plt.plot(Points0[:, 0], Points0[:, 1], '--')
#plt.plot(Points1[:, 0], Points1[:, 1], '-b')
#plt.plot(Points2[:, 0], Points2[:, 1], '-b')
#plt.show()
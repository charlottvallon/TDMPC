import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from Utilities import Curvature, nStepRegression
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from numpy import eye
# from osqp import OSQP

from abc import ABCMeta, abstractmethod
# import sys
# sys.path.append('../../SwitchSysLMPC/src')
# import pwa_cluster as pwac


solvers.options['show_progress'] = False

class AbstractControllerLMPC:
    __metaclass__ = ABCMeta
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """

    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        """Initialization
        Arguments:
            numSS_Points: number of points selected from the previous trajectories to build SS
            numSS_it: number of previois trajectories selected to build SS
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||x_{k+1}-x_k||_dR
            n,d: state and input dimensiton
            shift: given the closest point x_t^j to x(t) the controller start selecting the point for SS from x_{t+shift}^j
            track_map: track_map
            Laps: maximum number of laps the controller can run (used to avoid dynamic allocation)
            TimeLMPC: maximum time [s] that an lap can last (used to avoid dynamic allocation)
            Solver: solver used in the reformulation of the LMPC as QP
        """
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        self.N = N
        self.Qslack = Qslack
        self.Q = Q
        self.R = R
        self.dR = dR
        self.n = n
        self.d = d
        self.shift = shift
        self.dt = dt
        self.track_map = track_map
        self.Solver = Solver            
        self.clustering = None
        self.OldInput = np.zeros((1,d))

        # Initialize the following quantities to avoid dynamic allocation
        # TODO: is there a more graceful way to do this in python?
        NumPoints = int(TimeLMPC / dt) + 1
        self.TimeSS  = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS      = 10000 * np.ones((NumPoints, n, Laps))    # Sampled Safe SS
        self.uSS     = 10000 * np.ones((NumPoints, d, Laps))    # Input associated with the points in SS
        self.Qfun    =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        # TODO replace with after-the-fact mapping?
        self.SS_glob = 10000 * np.ones((NumPoints, n, Laps))    # SS in global (X-Y) used for plotting

        # Initialize the controller iteration
        self.it      = 0

        # Initialize pool for parallel computing used in the internal function _LMPC_EstimateABC
        # TODO this parameter should be tunable
        self.p = Pool(4)

    def solve(self, x0, uOld=np.zeros([0, 0])):
        """Computes control action
        Arguments:
            x0: current state position
        """

        # Select Points from Safe Set
        # a subset of nearby points are chosen from past iterations
        SS_PointSelectedTot      = np.empty((self.n, 0))
        Qfun_SelectedTot         = np.empty((0))
        for jj in range(0, self.numSS_it):
            SS_PointSelected, Qfun_Selected = SelectPoints(self.SS, self.Qfun, self.it - jj - 1, x0, self.numSS_Points / self.numSS_it, self.shift)
            SS_PointSelectedTot =  np.append(SS_PointSelectedTot, SS_PointSelected, axis=1)
            Qfun_SelectedTot    =  np.append(Qfun_SelectedTot, Qfun_Selected, axis=0)

        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot    = Qfun_SelectedTot

        # Get the matrices for defining the QP
        # this method will be defined in inheriting classes
        L, G, E, M, q, F, b = self._getQP(x0)
        
        # Solve QP
        startTimer = datetime.datetime.now()
        if self.Solver == "CVX":
            res_cons = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
            if res_cons['status'] == 'optimal':
                feasible = 1
            else:
                feasible = 0
            Solution = np.squeeze(res_cons['x'])     
        elif self.Solver == "OSQP":
            # Adaptation for QSQP from https://github.com/alexbuyval/RacingLMPC/
            res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add(np.dot(E,x0),L[:,0]))
            Solution = res_cons.x
        deltaTimer = datetime.datetime.now() - startTimer
        self.feasible = feasible
        self.solverTime = deltaTimer

        # Extract solution and set linearization points
        xPred, uPred, lambd, slack = LMPC_GetPred(Solution, self.n, self.d, self.N)
        #print uPred
        self.xPred = xPred.T
        if self.N == 1:
            self.uPred    = np.array([[uPred[0], uPred[1]]])
            self.LinInput =  np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], uPred.T[-1, :]))
        self.OldInput = uPred.T[0,:]
        # TODO: make this more general
        self.LinPoints = np.vstack((xPred.T[1:,:], xPred.T[-1,:]))

        # TODO: this is a temporary hack to store piecewise affine predictions
        # only putting this prediction in the one-step position
        if self.clustering is not None:
            pwa_pred = self.clustering.get_prediction(np.hstack([self.xPred[0],self.uPred[0]]))
            self.xPred[1] = pwa_pred
    
    def addReachableSet(self,rSS,ruSS,rQfun,shuffledMap):
        it = self.it
        end_it = rSS.shape[0]
        
        self.TimeSS[it] = end_it
        self.SS[0:end_it, :, it] = rSS
        
        rSS_glob = 1000*np.ones(rSS.shape)
        for state in range(rSS.shape[0]):
            rSS_glob[state,:] = rSS[state,:]
            global_values = shuffledMap.getGlobalPosition(rSS[state,4],rSS[state,5])
            rSS_glob[state,4] = global_values[0]
            rSS_glob[state,5] = global_values[1]
        self.SS_glob[0:end_it, :, it] = rSS_glob
               
        self.uSS[0:end_it, :, it] = ruSS
        self.Qfun[0:end_it, it] = rQfun
        
        if self.it == 0:
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]
            
        self.it = self.it + 1
        
    def addTrajectory(self, ClosedLoopData):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        it = self.it

        end_it = ClosedLoopData.SimTime
        self.TimeSS[it] = end_it
        self.SS[0:(end_it + 1), :, it] = ClosedLoopData.x[0:(end_it + 1), :]
        self.SS_glob[0:(end_it + 1), :, it] = ClosedLoopData.x_glob[0:(end_it + 1), :]
        self.uSS[0:end_it, :, it]      = ClosedLoopData.u[0:(end_it), :]
        self.Qfun[0:(end_it + 1), it]  = ComputeCost(ClosedLoopData.x[0:(end_it + 1), :],
                                                              ClosedLoopData.u[0:(end_it), :], self.track_map.TrackLength)
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1

        if self.it == 0:
            # TODO: made this more general
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]

        self.it = self.it + 1
        
    def addTrajectoryToSS(self, SafeSet,uSafeSet, qSafeSet, ClosedLoopData):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        it = self.it
            
        end_it = ClosedLoopData.SimTime
        self.TimeSS[it] = end_it
        SafeSet[0:(end_it + 1), :, it] = ClosedLoopData.x[0:(end_it + 1), :]
        #GlobalSafeSet[0:(end_it + 1), :, it] = ClosedLoopData.x_glob[0:(end_it + 1), :]
        uSafeSet[0:end_it, :, it]      = ClosedLoopData.u[0:(end_it), :]
        qSafeSet[0:(end_it + 1), it]  = ComputeCost(ClosedLoopData.x[0:(end_it + 1), :],
                                                              ClosedLoopData.u[0:(end_it), :], self.track_map.TrackLength)
        for i in np.arange(0, qSafeSet.shape[0]):
            if qSafeSet[i, it] == 0:
                qSafeSet[i, it] = qSafeSet[i - 1, it] - 1

        if self.it == 0:
            # TODO: made this more general
            self.LinPoints = SafeSet[1:self.N + 2, :, it]
            self.LinInput  = uSafeSet[1:self.N + 1, :, it]

        self.it = self.it + 1

    def addPoint(self, x, u, i):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter + i + 1, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.track_map.TrackLength, 0])
        self.uSS[Counter + i + 1, :, self.it - 1] = u
        if self.Qfun[Counter + i + 1, self.it - 1] == 0:
            self.Qfun[Counter + i + 1, self.it - 1] = self.Qfun[Counter + i, self.it - 1] - 1
        # TODO: this is a temporary hack to store piecewise affine predictions
        # won't work for more than one LMPC lap
        if self.clustering is not None:
            self._estimate_pwa(x, u)
    

    def update(self, SS, uSS, Qfun, TimeSS, it, LinPoints, LinInput):
        """update controller parameters. This function is useful to transfer information among LMPC controller
           with different tuning
        Arguments:
            SS: sampled safe set
            uSS: input associated with the points in SS
            Qfun: Qfun: cost-to-go from each point in SS
            TimeSS: time at which each j-th iteration is completed
            it: current iteration
            LinPoints: points used in the linearization and system identification procedure
            LinInput: inputs associated with the points used in the linearization and system identification procedure
        """
        self.SS  = SS
        self.uSS = uSS
        self.Qfun  = Qfun
        self.TimeSS  = TimeSS
        self.it = it

        self.LinPoints = LinPoints
        self.LinInput  = LinInput
        
    def splitTheSS(self, Map):        
        intervals = Map.PointAndTangent[:,3]
        num_modes = intervals.size
        #add fourth dimension to safe set
        self.splitSS = 1000*np.ones((self.SS.shape[0], self.SS.shape[1],self.SS.shape[2],num_modes))
        self.splituSS = 1000*np.ones((self.uSS.shape[0],self.uSS.shape[1],self.uSS.shape[2],num_modes))
        self.splitQfun = 1000*np.ones((self.Qfun.shape[0],self.Qfun.shape[1],num_modes))
        
        #populate splitSS with modes
        for state in range(0,self.SS.shape[0]):
            for lap in range(0,self.SS.shape[2]):
                pointInQ = self.SS[state,4,lap]
                while pointInQ > Map.TrackLength:
                    pointInQ -= Map.TrackLength
                
                whichMode = int(np.argwhere(pointInQ>=intervals)[-1])
                interm = self.SS[state,:,lap]
                interm[4] = pointInQ
                self.splitSS[state,:,lap,whichMode] = interm
                self.splituSS[state,:,lap,whichMode] = self.uSS[state,:,lap]
                self.splitQfun[state,lap,whichMode] = self.Qfun[state,lap]

    def relTheSplitSS(self,Map):
        intervals = Map.PointAndTangent[:,3]
        self.relSplitSS = 10000*np.ones(self.splitSS.shape)
        
        #each mode's s entries gets normalized wrt the starting interval
        for mode in range(0,self.splitSS.shape[3]):
            relStart = intervals[mode]
            for lap in range(0,self.splitSS.shape[2]):
                for state in range(0,self.splitSS.shape[0]):
                    # put condition to check if this is an arbitrary entry!
                    if self.splitSS[state,0,lap,mode] < 1000:
                        self.relSplitSS[state,:,lap,mode] = self.splitSS[state,:,lap,mode] - np.array([0, 0, 0, 0, relStart, 0])
                        # we do not need to relativize the input, obviously, so we continue using splituSS
                                         
    def makeShuffledSS(self,Map):
        intervals = Map.PointAndTangent[:,3]
        
        #self.shuffledSplitSS = 10000*np.ones(self.relSplitSS.shape)
        self.shuffledSS = 10000*np.ones(self.SS.shape)
        self.shuffleduSS = 10000*np.ones(self.uSS.shape)
        self.shuffledQfun = 10000*np.ones(self.Qfun.shape)
        
        #shuffle modes
        #shuffledSplitSS = self.relSplitSS[:,:,:,Map.modeOrder.append(self.relSplitSS.shape[3])]
        self.shuffledSplitSS = self.relSplitSS[:,:,:,Map.modeOrder]
        self.shuffledSplituSS = self.splituSS[:,:,:,Map.modeOrder]
        self.shuffledSplitQfun = self.splitQfun[:,:,Map.modeOrder]
        
        #shuffle modes and turn relative safe set into absolute coordinates again
        for mode in range(0,self.relSplitSS.shape[3]):
            relStart = intervals[mode]
            for lap in range(0,self.relSplitSS.shape[2]):
                for state in range(0,self.relSplitSS.shape[0]):
                    # put condition to check if this is an arbitrary entry!
                    if self.shuffledSplitSS[state,0,lap,mode] < 1000:
                        self.shuffledSplitSS[state,:,lap,mode] = self.shuffledSplitSS[state,:,lap,mode] + np.array([0, 0, 0, 0, relStart, 0])
        
        #append everything together again; just need to have fewer than 4000 samples given these numbers
        for lap in range(0,self.relSplitSS.shape[2]):
            counter = 0;
            for mode in range(0,self.relSplitSS.shape[3]):
                #add point
                for state in range(0,self.relSplitSS.shape[0]):
                    if self.shuffledSplitSS[state,0,lap,mode] < 1000:
                        self.shuffledSS[counter,:,lap] = self.shuffledSplitSS[state,:,lap,mode]
                        self.shuffleduSS[counter,:,lap] = self.shuffledSplituSS[state,:,lap,mode]
                        self.shuffledQfun[counter,lap] = self.shuffledSplitQfun[state,lap,mode]
                        counter += 1
    
    def processQfun(self):
        for lap in range(0,self.Qfun.shape[1]):
            for state in range(0,self.Qfun.shape[0]):
                if self.Qfun[state,lap]<=0:
                    self.Qfun[state,lap]=1000
                    
    def reorganizeReachableSafeSet(self):
        # reorganize reachableSS, reachableuSS, reachableQfun to start at the top of the matrix               
        # reachableSS:        
        for lap in range(self.reachableSS.shape[2]):
            tbc = self.reachableSS[:,:,lap]
            count = 0
            for state in range(tbc.shape[0]):
                if tbc[state,0]<1000:
                    tbc[count,:] = tbc[state,:]
                    count+=1                    
            while count < tbc.shape[0]:
                tbc[count,:] = 1000*np.ones((1,6))
                count+=1
            self.reachableSS[:,:,lap] = tbc
        
        # reachableuSS:
        for lap in range(self.reachableuSS.shape[2]):
            tbc = self.reachableuSS[:,:,lap]
            count = 0
            for state in range(tbc.shape[0]):
                if tbc[state,0]<1000:
                    tbc[count,:] = tbc[state,:]
                    count+=1                    
            while count < tbc.shape[0]:
                tbc[count,:] = 1000*np.ones((1,2))
                count+=1
            self.reachableuSS[:,:,lap] = tbc
        
        #reachableQfun:
        for lap in range(self.reachableQfun.shape[1]):
            tbc = self.reachableQfun[:,lap]
            count = 0
            for state in range(tbc.shape[0]):
                if tbc[state]<1000:
                    tbc[count] = tbc[state]
                    count+=1
            while count<len(tbc):
                tbc[count]=1000
                count+=1
            self.reachableQfun[:,lap] = tbc
        
        
    def reachabilityAnalysis(self,A,B,Qslack,N):
        local_tv_N = 5
        
        # reachableSplitSS is initialized to size of shuffledSplitSS
        reachableSplitSS = 1000*np.ones(self.shuffledSplitSS.shape)
        reachableSplituSS = 1000*np.ones(self.shuffledSplituSS.shape)
        reachableSplitQfun = 1000*np.ones(self.shuffledSplitQfun.shape)
        
        # last mode (10) of reachableSplitSS is initialized to shuffledSplitSS(:,:,:,10)
        reachableSplitSS[:,:,:,-1] = self.shuffledSplitSS[:,:,:,-1]
        reachableSplituSS[:,:,:,-1] = self.shuffledSplituSS[:,:,:,-1]
        reachableSplitQfun[:,:,-1] = self.initializeShuffledCost(self.shuffledSplitQfun[:,:,-1]) #all entries[0] over all laps[1]
        # this works, verified
                
        # moving backwards through the shuffled modes
        for mode_iter in range(2,self.shuffledSplitSS.shape[3]+1):
            mode = self.shuffledSplitSS.shape[3]-mode_iter            
            #print 'Trying to connect modes ',mode,' and ', mode+1
            modeConnectionFound = False
            #minCost = 10000;
            minSlack = 0.4;
            
            # iterating through each lap
            for lap in range(0,self.it):
                array = self.shuffledSplitSS[:,4,lap,mode]
                startpoint_index = np.argmax(np.multiply(array,array<1000))
                startpoint = self.shuffledSplitSS[startpoint_index,:,lap,mode]   
                
                try:
                    self.LinPoints = np.vstack((self.shuffledSplitSS[startpoint_index - local_tv_N  : startpoint_index,:,lap,mode]))
                    self.LinInput = np.vstack((self.shuffledSplituSS[startpoint_index - local_tv_N : startpoint_index - 1,:,lap,mode]))
                    #temporarily reset for estimateABC function
                    self.N = self.LinInput.shape[0]
                    
                    
                    locAvec, locBvec, locCvec, indexUsed_list = self._EstimateABC()
                    locA = locAvec[-1]
                    locB = locBvec[-1]
                    locC = locCvec[-1]
                except ValueError:
                    locA = A
                    locB = B
                    locC = np.zeros((6,1))
                                
                self.N = N
                
                lapConnectionFound = False
                
                # iterate through the first points of the laps in the next shuffled mode
                for cont_lap in range(0,self.it):
                    #print "cont_lap ", cont_lap
                    array = reachableSplitSS[:,:,cont_lap,mode+1]
                    try:
                        BigZ_array = array[np.hstack(np.argwhere(array[:,1]<100))]
                        BigQ_array = reachableSplitQfun[np.hstack(np.argwhere(array[:,1]<100)),cont_lap,mode+1]
                    except ValueError:
                        #print "Value Error"
                        continue
                    
                    status, ninput, cost, slacknorm = self.isReachable(startpoint, BigZ_array, BigQ_array, locA, locB, locC, Qslack)
                    
                    if status and slacknorm < minSlack:
                        #print 'Lap ', lap, ' connected to lap ', cont_lap
                        lapConnectionFound = True
                        modeConnectionFound = True
                        bestNextLap = cont_lap
                        
                        minSlack = slacknorm
                        reachableSplitSS[:,:,lap,mode] = self.shuffledSplitSS[:,:,lap,mode]

                        inputMatrixSize = np.argwhere(self.shuffledSplituSS[:,0,lap,mode]<1000)                            
                        reachableSplituSS[:,:,lap,mode] = self.shuffledSplituSS[:,:,lap,mode]
                        reachableSplituSS[inputMatrixSize,:,lap,mode] = ninput
                        reachableSplitQfun[:,lap,mode] = self.updateShuffledCost(self.shuffledSplitQfun[:,lap,mode], reachableSplitQfun[:,cont_lap,mode+1])        
                        #print 'Cost vector: ', reachableSplitQfun[reachableSplitQfun[:,lap,mode]<1000,lap,mode]
                        
                if lapConnectionFound:
                    for fixMode in range(mode+1,self.shuffledSplitSS.shape[3]):
                        reachableSplitSS[:,:,lap,fixMode] = reachableSplitSS[:,:,bestNextLap,fixMode]
                        reachableSplituSS[:,:,lap,fixMode] = reachableSplituSS[:,:,bestNextLap,fixMode]
                        reachableSplitQfun[:,lap,fixMode] = reachableSplitQfun[:,bestNextLap,fixMode]   
                else: 
                    # delete the numbers for that lap (implemented by setting to 10000)
                    reachableSplitSS[:,:,lap,mode] = 1000*np.ones((reachableSplitSS.shape[0],reachableSplitSS.shape[1]))   
                    
            if modeConnectionFound:
                print 'Connected modes ', mode, ' and ', mode+1
            else: 
                print 'Could not connect modes ', mode, ' and ', mode+1, '. Exiting.'
                return 
               
        # reachableSS is initialized to size of shuffledSS
        reachableSS = 1000*np.ones(self.shuffledSS.shape)
        reachableuSS = 1000*np.ones(self.shuffleduSS.shape)
        reachableQfun = 1000*np.ones(self.shuffledQfun.shape)
            
        # recombine reachableSplitSS into reachableSS! 
        for lap in range(0,reachableSplitSS.shape[2]):
            counter = 0;
            for mode in range(0,reachableSplitSS.shape[3]):
                #add point
                for state in range(0,reachableSplitSS.shape[0]):
                    if reachableSplitSS[state,0,lap,mode] < 1000:
                        reachableSS[counter,:,lap] = reachableSplitSS[state,:,lap,mode]
                        reachableuSS[counter,:,lap] = reachableSplituSS[state,:,lap,mode]
                        reachableQfun[counter,lap] = reachableSplitQfun[state,lap,mode]
                        counter += 1
                       
        self.reachableSplitSS = reachableSplitSS
        self.reachableSplituSS = reachableSplituSS
        self.reachableSplitQfun = reachableSplitQfun
        self.reachableSS = reachableSS
        self.reachableuSS = reachableuSS
        self.reachableQfun = reachableQfun
        
        
    def initializeShuffledCost(self,Qarray):
        resetQarray = 1000*np.ones(Qarray.shape)
        for lap in range(Qarray.shape[1]):
            num_steps_in_mode = sum(Qarray[:,lap]<1000)
            step_indices = np.argwhere(Qarray[:,lap]<1000)
            resetQarray[step_indices,lap] = np.arange(num_steps_in_mode,0,-1).reshape((-1,1))           
        return resetQarray
    
    def updateShuffledCost(self, Qvec, nextModeQvec):
        # these are vectors, no longer arrays as in initializeShuffledCost
        resetQvec = 1000*np.ones(Qvec.shape)
        num_steps_in_mode = sum(Qvec<1000)
        step_indices = np.argwhere(Qvec<1000)
        
        # cost of next step
        start_cost = max(nextModeQvec[nextModeQvec<1000])
        
        resetQvec[step_indices] = start_cost + np.arange(num_steps_in_mode,0,-1).reshape((-1,1))
        
        return resetQvec
        
    def isReachable(self, x0, BigZ, BigQ, A, B, C, Qslack):
        
        M, q, F, b, G, d = self.BuildReachableQP(x0, BigZ, BigQ, A, B, C, Qslack)
        sol = qp(matrix(M), matrix(q), matrix(F), matrix(b), matrix(G), matrix(d))
        if sol['status'] == 'optimal':
            status = True
        else:
            status = False    
        Solution = np.squeeze(sol['x']) 
        
        # need to return: status, input, optimization cost
        ninput = [Solution[-3], Solution[-2]]
        slack_variable = Solution[-9:-4]
        slacknorm = np.linalg.norm(slack_variable)
        # extract cost associated with solution?
        cost = sol['primal objective']
        
        return status, ninput, cost, slacknorm
    
    def BuildReachableQP(self, x0, BigZ, BigQ, A, B, C, Qslack):
        num_x = A.shape[0]
        num_u = B.shape[1]
        num_lam = BigQ.shape[0]
        size_z = 3*num_x + num_lam + num_u + 1        
        
        #max_slack_value = 0.05 
        
        # Cost matrices M, q: 
        #x0, xf
        M1 = 0*np.eye(A.shape[0])
        #xf
        M2 = 0*np.eye(A.shape[0])
        M3 = 0*np.eye(num_lam)
        #M4 = Qslack
        #u and xc
        M5 = np.zeros((num_u+1,num_u+1))
        # append
        M = linalg.block_diag(M1, M2, M3, 200*Qslack, M5)
        
        q = np.zeros((size_z,1))
        q[-1] = 2
        
        # Inequality constraints (on input and slack variable) F, b: 
        Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])
        bu = np.array([[0.5],  # Max Steering
                   [0.5],  # Max Steering
                   [1.],  # Max Acceleration
                   [1.]])    
        F1 = np.hstack((np.zeros((4,3*num_x+num_lam)),Fu,np.zeros((4,1))))
        F2 = np.hstack((np.zeros((num_lam, 2*num_x)), -np.eye(num_lam), np.zeros((num_lam, num_x + num_u + 1)) ))
        bs = np.zeros((num_lam,1))
        F = np.vstack((F1, F2))
        b = np.vstack((bu, bs))
        
        # Equality constraints G, d:
        #print C.shape
        #print np.transpose(x0).shape
        #print np.zeros((num_x,1)).shape
        d = np.vstack((C, np.transpose(x0).reshape((num_x,1)), np.zeros((num_x,1)), [[0]], [[1]]))
        
        G1 = np.hstack(( -A, np.eye(6), np.zeros((num_x, num_x + num_lam)), -B, np.zeros((num_x,1)) ))
        G2 = np.hstack((np.eye(num_x), np.zeros((num_x, size_z - num_x)) ))
        G3 = np.hstack(( np.zeros((num_x, num_x)), np.eye(num_x), -np.transpose(BigZ), np.eye(num_x), np.zeros((num_x, num_u+1)) ))       
        G4 = np.hstack(( np.zeros((1,2*num_x)), -BigQ.reshape((1,num_lam)), np.zeros((1,num_x + num_u)), [[1]]))        
        G5 = np.hstack(( np.zeros((1,2*num_x)), np.ones((1,num_lam)), np.zeros((1, num_x + num_u + 1)) ))
        
        G = np.vstack((G1,G2,G3,G4,G5))     
        
        return M, q, F, b, G, d
        
    def selectBestTrajectory(self):
        SS = self.reachableSS
        uSS = self.reachableuSS
        Qfun = self.reachableQfun
        
        best_lap_candidates = np.argwhere(SS[0,4,:]<0.2)
        best_Q_of_lap_candidates = np.argmin(Qfun[0,best_lap_candidates])
        best_lap_index = int(best_lap_candidates[best_Q_of_lap_candidates])
    
        rSS = SS[:,:,best_lap_index]
        ruSS = uSS[:,:,best_lap_index]
        rQfun = Qfun[:,best_lap_index]
        
        return rSS, ruSS, rQfun

class PWAControllerLMPC(AbstractControllerLMPC):
    """
    Piecewise affine controller
    For now, uses LTV LMPC control, but stores predictions from a piecewise affine model
    """

    def __init__(self, n_clusters, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        # Build matrices for inequality constraints
        self.F, self.b = LMPC_BuildMatIneqConst(N, n, numSS_Points, Solver)
        self.n_clusters = n_clusters
        # python 2/3 compatibility
        super(PWAControllerLMPC, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
    
    def _getQP(self, x0):
        # Run System ID
        startTimer = datetime.datetime.now()
        Atv, Btv, Ctv, _ = self._EstimateABC()
        deltaTimer = datetime.datetime.now() - startTimer
        L, npG, npE = BuildMatEqConst_TV(self.Solver, Atv, Btv, Ctv)
        self.linearizationTime = deltaTimer

        # PWA System ID
        self._estimate_pwa(verbose=True)


        # Build Terminal cost and Constraint
        G, E = LMPC_TermConstr(self.Solver, self.N, self.n, self.d, npG, npE, self.SS_PointSelectedTot)
        M, q = LMPC_BuildMatCost(self.Solver, self.N, self.Qfun_SelectedTot, self.numSS_Points, self.Qslack, self.Q, self.R, self.dR, self.OldInput)
        return L, G, E, M, q, self.F, self.b

    def _EstimateABC(self):
        LinPoints       = self.LinPoints
        LinInput        = self.LinInput
        N               = self.N
        n               = self.n
        d               = self.d
        SS              = self.SS
        uSS             = self.uSS
        TimeSS          = self.TimeSS
        PointAndTangent = self.track_map.PointAndTangent
        dt              = self.dt
        it              = self.it
        p               = self.p

        ParallelComputation = 0 # TODO
        Atv = []; Btv = []; Ctv = []; indexUsed_list = []

        usedIt = range(it-2,it)
        MaxNumPoint = 40  # TODO Need to reason on how these points are selected

        if ParallelComputation == 1:
            # Parallel Implementation
            Fun = partial(RegressionAndLinearization, LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                           MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt)

            index = np.arange(0, N)  # Create the index vector

            Res = p.map(Fun, index)  # Run the process in parallel
            ParallelResutl = np.asarray(Res)

        for i in range(0, N):
            if ParallelComputation == 0:
               Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                                   MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i)
               Atv.append(Ai); Btv.append(Bi); Ctv.append(Ci)
               indexUsed_list.append(indexSelected)
            else:
               Atv.append(ParallelResutl[i][0])
               Btv.append(ParallelResutl[i][1])
               Ctv.append(ParallelResutl[i][2])
               indexUsed_list.append(ParallelResutl[i][3])

        return Atv, Btv, Ctv, indexUsed_list

    def _estimate_pwa(self, x=None, u=None, verbose=False):
        if self.clustering is None:
            # construct z and y from past laps
            zs = []; ys = []
            for it in range(self.it-1):
                states = self.SS[:int(self.TimeSS[it]), :, it]
                inputs = self.uSS[:int(self.TimeSS[it]), :, it]
                zs.append(np.hstack([states[:-1], inputs[:-1]]))
                ys.append(states[1:])
            zs = np.squeeze(np.array(zs)); ys = np.squeeze(np.array(ys))
            self.clustering = pwac.ClusterPWA.from_num_clusters(zs, ys, 
                                    self.n_clusters, z_cutoff=self.n)
            self.clustering.fit_clusters(verbose=verbose)
            # self.clustering.determine_polytopic_regions(verbose=verbose)



class ControllerLMPC(AbstractControllerLMPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """
    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        # Build matrices for inequality constraints
        self.F, self.b = LMPC_BuildMatIneqConst(N, n, numSS_Points, Solver)
        super(ControllerLMPC, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
    

    def _getQP(self, x0):
        # Run System ID
        A = np.array([[ 9.66986188e-01,  8.40031817e-02, -1.52758633e-02, 3.02222257e-02,  7.41213685e-05, -1.79188470e-02],
                      [-1.70819841e-03, -2.80969448e-02,  7.98021057e-03, -2.71158860e-03, -4.05797719e-06,  3.87380046e-03],
                      [-2.82696677e-02, -1.81939014e+00,  2.38112509e-01,-4.58330442e-02, -9.49417969e-05,  6.80457566e-02],
                      [ 1.34474870e-02, -1.84415868e-01,  4.17324683e-02,1.07733934e+00,  5.57120133e-05,  1.49890698e-02],
                      [ 9.76859874e-02, -2.35704565e-02,  2.41429308e-03,-4.68612085e-03,  9.99999205e-01, -2.09009762e-02],
                      [ 7.04133161e-04, -1.73933509e-02,  6.01945948e-03,6.83918582e-02,  3.55771523e-06,  1.00026000e+00]])
    
        B = np.array([[ 2.62522372e-02,  9.94735231e-02],
                      [ 2.56478316e-01,  3.71261813e-03],
                      [ 2.46539680e+00,  5.84412973e-02],
                      [ 1.75961692e-01,  1.60220732e-02],
                      [ 1.51077144e-03,  4.14928905e-03],
                      [ 2.29064500e-02, -1.32228500e-04]])
        
        
        startTimer = datetime.datetime.now()
        try:
            Atv, Btv, Ctv, _ = self._EstimateABC()
        except ValueError:
            # approximate with regular A
            Atv = []; Btv = [];Ctv = []
            C = np.zeros((A.shape[0],1))
            for i in range(0, self.N):
               Atv.append(A); Btv.append(B); Ctv.append(C)
        deltaTimer = datetime.datetime.now() - startTimer
        L, npG, npE = BuildMatEqConst_TV(self.Solver, Atv, Btv, Ctv)
        self.linearizationTime = deltaTimer

        # Build Terminal cost and Constraint
        G, E = LMPC_TermConstr(self.Solver, self.N, self.n, self.d, npG, npE, self.SS_PointSelectedTot)
        M, q = LMPC_BuildMatCost(self.Solver, self.N, self.Qfun_SelectedTot, self.numSS_Points, self.Qslack, self.Q, self.R, self.dR, self.OldInput)
        return L, G, E, M, q, self.F, self.b

    def _EstimateABC(self):
        LinPoints       = self.LinPoints
        LinInput        = self.LinInput
        N               = self.N
        n               = self.n
        d               = self.d
        SS              = self.SS
        uSS             = self.uSS
        TimeSS          = self.TimeSS
        PointAndTangent = self.track_map.PointAndTangent
        dt              = self.dt
        it              = self.it
        p               = self.p

        ParallelComputation = 0 # TODO
        Atv = []; Btv = []; Ctv = []; indexUsed_list = []

        usedIt = range(it-2,it)
        MaxNumPoint = 40  # TODO Need to reason on how these points are selected

        if ParallelComputation == 1:
            # Parallel Implementation
            Fun = partial(RegressionAndLinearization, LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                           MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt)

            index = np.arange(0, N)  # Create the index vector

            Res = p.map(Fun, index)  # Run the process in parallel
            ParallelResutl = np.asarray(Res)

        for i in range(0, N):
            if ParallelComputation == 0:
               Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                                   MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i)
               Atv.append(Ai); Btv.append(Bi); Ctv.append(Ci)
               indexUsed_list.append(indexSelected)
            else:
               Atv.append(ParallelResutl[i][0])
               Btv.append(ParallelResutl[i][1])
               Ctv.append(ParallelResutl[i][2])
               indexUsed_list.append(ParallelResutl[i][3])

        return Atv, Btv, Ctv, indexUsed_list


class ControllerLTI_LMPC(AbstractControllerLMPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """
    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        # Build matrices for inequality constraints
        self.F, self.b = LMPC_BuildMatIneqConst(N, n, numSS_Points, Solver)
        super(ControllerLTI_LMPC, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
        
        self.A = np.array([[ 9.64629458e-01,  4.60237683e-01, -5.25408848e-02, 9.24358247e-04,  5.10264151e-05,  6.86962525e-03],
                           [-1.77378424e-03, -2.12240216e-01,  2.68469608e-02, -3.19560445e-04,  9.70106541e-06,  2.86591934e-03],
                           [-3.07763289e-02, -4.14081475e+00,  4.74387233e-01, -8.57617964e-03,  1.97579640e-04,  2.69366065e-02],
                           [ 2.08412726e-02, -7.59644071e-01,  1.01310692e-01, 1.08305080e+00, -1.87165855e-05,  5.04038912e-02],
                           [ 9.57579461e-02, -7.64598742e-03,  8.30621775e-04, -7.61048122e-03,  1.00000160e+00, -1.51358758e-02],
                           [ 9.35746415e-04, -1.04358461e-02,  5.34305392e-03, 6.87529950e-02, -2.19782880e-06,  1.00148367e+00]])
    
        self.B = np.array([[ 1.88757665e-02,  1.03291797e-01],
                           [ 2.59402708e-01,  1.83342143e-03],
                           [ 2.50156745e+00,  3.76333055e-02],
                           [ 1.82673285e-01, -7.43195820e-03],
                           [ 5.79452975e-04,  6.05520413e-03],
                           [ 2.34122782e-02, -9.64502589e-04]])

    def _getQP(self, x0):
                
        startTimer = datetime.datetime.now()
        
        Atv = []; Btv = [];Ctv = []
        C = np.zeros((self.A.shape[0],1))
        for i in range(0, self.N):
            Atv.append(self.A); Btv.append(self.B); Ctv.append(C)
        deltaTimer = datetime.datetime.now() - startTimer
        L, npG, npE = BuildMatEqConst_TV(self.Solver, Atv, Btv, Ctv)
        self.linearizationTime = deltaTimer

        # Build Terminal cost and Constraint
        G, E = LMPC_TermConstr(self.Solver, self.N, self.n, self.d, npG, npE, self.SS_PointSelectedTot)
        M, q = LMPC_BuildMatCost(self.Solver, self.N, self.Qfun_SelectedTot, self.numSS_Points, self.Qslack, self.Q, self.R, self.dR, self.OldInput)
        return L, G, E, M, q, self.F, self.b
    
class ControllerLTI_LMPC_NStep(AbstractControllerLMPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """
    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        # Build matrices for inequality constraints
        self.F, self.b = LMPC_BuildMatIneqConst(N, n, numSS_Points, Solver)
        super(ControllerLTI_LMPC_NStep, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
        
        self.A = np.array([[ 9.64629458e-01,  4.60237683e-01, -5.25408848e-02, 9.24358247e-04,  5.10264151e-05,  6.86962525e-03],
                           [-1.77378424e-03, -2.12240216e-01,  2.68469608e-02, -3.19560445e-04,  9.70106541e-06,  2.86591934e-03],
                           [-3.07763289e-02, -4.14081475e+00,  4.74387233e-01, -8.57617964e-03,  1.97579640e-04,  2.69366065e-02],
                           [ 2.08412726e-02, -7.59644071e-01,  1.01310692e-01, 1.08305080e+00, -1.87165855e-05,  5.04038912e-02],
                           [ 9.57579461e-02, -7.64598742e-03,  8.30621775e-04, -7.61048122e-03,  1.00000160e+00, -1.51358758e-02],
                           [ 9.35746415e-04, -1.04358461e-02,  5.34305392e-03, 6.87529950e-02, -2.19782880e-06,  1.00148367e+00]])
    
        self.B = np.array([[ 1.88757665e-02,  1.03291797e-01],
                           [ 2.59402708e-01,  1.83342143e-03],
                           [ 2.50156745e+00,  3.76333055e-02],
                           [ 1.82673285e-01, -7.43195820e-03],
                           [ 5.79452975e-04,  6.05520413e-03],
                           [ 2.34122782e-02, -9.64502589e-04]])

    def _getQP(self, x0):
                
        startTimer = datetime.datetime.now()        
        
#        Atv = []; Btv = [];Ctv = []
#        C = np.zeros((self.A.shape[0],1))
#        for i in range(0, self.N):
#            Atv.append(self.A); Btv.append(self.B); Ctv.append(C)
            
        deltaTimer = datetime.datetime.now() - startTimer
        
        # do the n-step regression here to get the Theta values?
        #Theta = self.nStepTheta
        L, npG, npE = self._BuildMatEqConst_Nstep(self.Solver, self.Theta, self.N, self.n, self.d)
        
        self.linearizationTime = deltaTimer

        # Build Terminal cost and Constraint
        G, E = LMPC_TermConstr(self.Solver, self.N, self.n, self.d, npG, npE, self.SS_PointSelectedTot)
        M, q = LMPC_BuildMatCost(self.Solver, self.N, self.Qfun_SelectedTot, self.numSS_Points, self.Qslack, self.Q, self.R, self.dR, self.OldInput)
        return L, G, E, M, q, self.F, self.b
    
    def _getTheta(self, x, u, N, lamb):
        
        Theta, nStep_Error = nStepRegression(x, u, N, lamb)
        self.Theta = Theta

    def _BuildMatEqConst_Nstep(self, Solver, Theta, N, n, d):
    
        # dimensions of these matrices stay the same
        Gx = np.eye(n * (N + 1))
        Gu = np.zeros((n * (N + 1), d * (N)))
    
        E = np.zeros((n * (N + 1), n))
        E[np.arange(n)] = np.eye(n)
    
        # L stays all zeros except end
        L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
        L[-1] = 1 # Summmation of lambda must add up to 1
        
        # Theta_x(n) always acts on x0, but Theta_x(n) changes with n
        ind2x = np.arange(n)
        thetaXind = np.array(xrange(1))
        thetaUind = np.array(xrange(1))
   
        for i in range(0, N):
            # starts by defining equation for x1=Theta1[x0,u0]. x2 = Theta2[x0,u0,u1], etc...
            
            # select Theta indices
            if i==0:                
                thetaXind = thetaUind[-1] + np.arange(n)
                
            else:
                thetaXind = thetaUind[-1] + 1 + np.arange(n)
                
            thetaUind = thetaXind[-1] + 1 + np.arange((i+1)*d)
            
            # vertical index range is 6 (state length, stays the same)
            ind1 = n + i * n + np.arange(n)
            Gx[np.ix_(ind1, ind2x)] = -Theta[:,thetaXind]            
            
            # UPDATE WHICH INDICES IN GU CORRESPOND TO THE THETA HERE
            ind2u = np.arange((i+1)*d)
            Gu[np.ix_(ind1, ind2u)] = -Theta[:, thetaUind]
    
        G = np.hstack((Gx, Gu))
    
    
        if Solver == "CVX":
            L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0].astype(int), np.nonzero(L)[1].astype(int), L.shape)
            L_return = L_sparse
        else:
            L_return = L
        
        return L_return,G,E
    

# ======================================================================================================================
# ======================================================================================================================
# =============================== Utility functions for LMPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================

def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    osqp = OSQP()
    if G is not None:
        l = -inf * ones(len(h))
        if A is not None:
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l, b])
            qp_u = hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible

def LMPC_BuildMatCost(Solver, N, Sel_Qfun, numSS_Points, Qslack, Q, R, dR, uOld):
    n = Q.shape[0]
    P = Q
    vt = 2

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R + 2*np.diag(dR)] * (N)

    Mu = linalg.block_diag(*c)
    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)
    # np.savetxt('Mu.csv', Mu, delimiter=',', fmt='%f')

    M00 = linalg.block_diag(Mx, P, Mu)
    M0 = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q0 = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)

    # Derivative Input
    q0[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

    # np.savetxt('q0.csv', q0, delimiter=',', fmt='%f')
    q = np.append(np.append(q0, Sel_Qfun), np.zeros(Q.shape[0]))

    # np.savetxt('q.csv', q, delimiter=',', fmt='%f')

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    if Solver == "CVX":
        M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
        M_return = M_sparse
    else:
        M_return = M

    return M_return, q

def LMPC_BuildMatIneqConst(N, n, numSS_Points, solver):
    # Build the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[3.],  # vx max
                   [0.8],  # max ey
                   [0.8]])  # max ey

    # Build the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.5],  # Max Steering
                   [0.5],  # Max Steering
                   [1.],  # Max Acceleration
                   [1.]])  # Max Acceleration

    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))

    FDummy = np.vstack((Dummy1, Dummy2))
    I = -np.eye(numSS_Points)
    FDummy2 = linalg.block_diag(FDummy, I)
    Fslack = np.zeros((FDummy2.shape[0], n))
    F = np.hstack((FDummy2, Fslack))

    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')
    b = np.hstack((bxtot, butot, np.zeros(numSS_Points)))
    if solver == "CVX":
        F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
        F_return = F_sparse
    else:
        F_return = F

    return F_return, b


def SelectPoints(SS, Qfun, it, x0, numSS_Points, shift):
    # selects the closest point in the safe set to x0
    # returns a subset of the safe set which contains a range of points ahead of this point
    x = SS[:, :, it]
    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    MinNorm = np.argmin(norm)

    if (MinNorm + shift >= 0):
        # TODO: what if shift + MinNorm + numSS_Points is greater than the points in the safe set?
        SS_Points = x[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), it]
    else:
        SS_Points = x[int(MinNorm):int(MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(MinNorm):int(MinNorm + numSS_Points), it]

    return SS_Points, Sel_Qfun

def ComputeCost(x, u, TrackLength):
    Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, x.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            Cost[x.shape[0] - 1 - i] = 0
        elif x[x.shape[0] - 1 - i, 4]< TrackLength:
            Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
        else:
            Cost[x.shape[0] - 1 - i] = 0

    return Cost


def LMPC_TermConstr(Solver, N, n, d, G, E, SS_Points):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

    TermCons = np.zeros((n, (N + 1) * n + N * d))
    TermCons[:, N * n:(N + 1) * n] = np.eye(n)

    G_enlarged = np.vstack((G, TermCons))

    G_lambda = np.zeros(( G_enlarged.shape[0], SS_Points.shape[1] + n))
    G_lambda[G_enlarged.shape[0] - n:G_enlarged.shape[0], :] = np.hstack((-SS_Points, np.eye(n)))

    G_LMPC0 = np.hstack((G_enlarged, G_lambda))
    G_ConHull = np.zeros((1, G_LMPC0.shape[1]))
    G_ConHull[-1, G_ConHull.shape[1]-SS_Points.shape[1]-n:G_ConHull.shape[1]-n] = np.ones((1,SS_Points.shape[1]))

    G_LMPC = np.vstack((G_LMPC0, G_ConHull))

    E_LMPC = np.vstack((E, np.zeros((n + 1, n))))

    # np.savetxt('G.csv', G_LMPC, delimiter=',', fmt='%f')
    # np.savetxt('E.csv', E_LMPC, delimiter=',', fmt='%f')

    if Solver == "CVX":
        G_LMPC_sparse = spmatrix(G_LMPC[np.nonzero(G_LMPC)], np.nonzero(G_LMPC)[0].astype(int), np.nonzero(G_LMPC)[1].astype(int), G_LMPC.shape)
        E_LMPC_sparse = spmatrix(E_LMPC[np.nonzero(E_LMPC)], np.nonzero(E_LMPC)[0].astype(int), np.nonzero(E_LMPC)[1].astype(int), E_LMPC.shape)
        G_LMPC_return = G_LMPC_sparse
        E_LMPC_return = E_LMPC_sparse
    else:
        G_LMPC_return = G_LMPC
        E_LMPC_return = E_LMPC

    return G_LMPC_return, E_LMPC_return

def BuildMatEqConst_TV(Solver, A, B, C):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    N = len(A)
    n, d = B[0].shape
    #n,d = 6,2
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
    L[-1] = 1 # Summmation of lamba must add up to 1

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))


    if Solver == "CVX":
        L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0].astype(int), np.nonzero(L)[1].astype(int), L.shape)
        L_return = L_sparse
    else:
        L_return = L

    return L_return, G, E

def LMPC_GetPred(Solution,n,d,N):
    # logic to decompose the QP solution
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    lambd = Solution[n*(N+1)+d*N:Solution.shape[0]-n]
    slack = Solution[Solution.shape[0]-n:]
    return xPred, uPred, lambd, slack

# ======================================================================================================================
# ======================================================================================================================
# ========================= Utility functions for Local Regression and Linearization ===================================
# ======================================================================================================================
# ======================================================================================================================


def RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS, MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i):
    x0 = LinPoints[i, :]

    Ai = np.zeros((n, n))
    Bi = np.zeros((n, d))
    Ci = np.zeros((n, 1))

    # Compute Index to use
    h = 5
    lamb = 0.0
    stateFeatures = [0, 1, 2]
    ConsiderInput = 1

    if ConsiderInput == 1:
        scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        xLin = np.hstack((LinPoints[i, stateFeatures], LinInput[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        xLin = LinPoints[i, stateFeatures]

    indexSelected = []
    K = []
    for i in usedIt:
        indexSelected_i, K_i = ComputeIndex(h, SS, uSS, TimeSS, i, xLin, stateFeatures, scaling, MaxNumPoint,
                                            ConsiderInput)
        indexSelected.append(indexSelected_i)
        K.append(K_i)

    # =========================
    # ====== Identify vx ======
    inputFeatures = [1]
    Q_vx, M_vx = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 0
    b = Compute_b(SS, yIndex, usedIt, matrix, M_vx, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_vx, b, stateFeatures,
                                                                                      inputFeatures, qp)

    # =======================================
    # ====== Identify Lateral Dynamics ======
    inputFeatures = [0]
    Q_lat, M_lat = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 1  # vy
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)

    yIndex = 2  # wz
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)

    # ===========================
    # ===== Linearization =======
    vx = x0[0]; vy   = x0[1]
    wz = x0[2]; epsi = x0[3]
    s  = x0[4]; ey   = x0[5]

    if s < 0:
        print("s is negative, here the state: \n", LinPoints)

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    cur = Curvature(s, PointAndTangent)
    cur = Curvature(s, PointAndTangent)
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3] = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur ) - np.dot(Ai[3, :], x0)

    # ===========================
    # ===== Linearize s =========
    # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
    ds_vx = dt * (np.cos(epsi) / den)
    ds_vy = -dt * (np.sin(epsi) / den)
    ds_wz = 0
    ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
    ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
    ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)

    Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
    Ci[4] = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) ) - np.dot(Ai[4, :], x0)

    # ===========================
    # ===== Linearize ey ========
    # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    dey_vx = dt * np.sin(epsi)
    dey_vy = dt * np.cos(epsi)
    dey_wz = 0
    dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
    dey_s = 0
    dey_ey = 1

    Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
    Ci[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)

    deltaTimer_tv = datetime.datetime.now() - startTimer

    return Ai, Bi, Ci, indexSelected

def Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K):
    Counter = 0
    it = 1
    X0   = np.empty((0,len(stateFeatures)+len(inputFeatures)))
    Ktot = np.empty((0))

    for it in usedIt:
        X0 = np.append( X0, np.hstack((np.squeeze(SS[np.ix_(indexSelected[Counter], stateFeatures, [it])]),
                            np.squeeze(uSS[np.ix_(indexSelected[Counter], inputFeatures, [it])], axis=2))), axis=0)
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))


    return Q, M

def Compute_b(SS, yIndex, usedIt, matrix, M, indexSelected, K, np):
    Counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))

    for it in usedIt:
        y = np.append(y, np.squeeze(SS[np.ix_(indexSelected[Counter] + 1, [yIndex], [it])]))
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))

    return b

def LMPC_LocLinReg(Q, b, stateFeatures, inputFeatures, qp):
    import numpy as np
    from numpy import linalg as la
    import datetime

    # K = np.ones(len(index))

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    res_cons = qp(Q, b) # This is ordered as [A B C]

    deltaTimer_tv = datetime.datetime.now() - startTimer

    # print("Non removable time: ", deltaTimer_tv.total_seconds())
    Result = np.squeeze(np.array(res_cons['x']))
    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C

def ComputeIndex(h, SS, uSS, TimeSS, it, x0, stateFeatures, scaling, MaxNumPoint, ConsiderInput):
    import numpy as np
    from numpy import linalg as la
    import datetime



    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    oneVec = np.ones( (SS[0:TimeSS[it], :, it].shape[0]-1, 1) )

    x0Vec = (np.dot( np.array([x0]).T, oneVec.T )).T

    if ConsiderInput == 1:
        DataMatrix = np.hstack((SS[0:TimeSS[it]-1, stateFeatures, it], uSS[0:TimeSS[it]-1, :, it]))
    else:
        DataMatrix = SS[0:TimeSS[it]-1, stateFeatures, it]

    # np.savetxt('A.csv', SS[0:TimeSS[it]-1, stateFeatures, it], delimiter=',', fmt='%f')
    # np.savetxt('B.csv', SS[0:TimeSS[it], :, it][0:-1, stateFeatures], delimiter=',', fmt='%f')
    # np.savetxt('SS.csv', SS[0:TimeSS[it], :, it], delimiter=',', fmt='%f')

    diff  = np.dot(( DataMatrix - x0Vec ), scaling)
    # print 'x0Vec \n',x0Vec
    norm = la.norm(diff, 1, axis=1)
    indexTot =  np.squeeze(np.where(norm < h))
    # print indexTot.shape, np.argmin(norm), norm, x0
    try:
        if (indexTot.shape[0] >= MaxNumPoint):
            index = np.argsort(norm)[0:MaxNumPoint]
            # MinNorm = np.argmin(norm)
            # if MinNorm+MaxNumPoint >= indexTot.shape[0]:
            #     index = indexTot[indexTot.shape[0]-MaxNumPoint:indexTot.shape[0]]
            # else:
            #     index = indexTot[MinNorm:MinNorm+MaxNumPoint]
        else:
            index = indexTot
    except IndexError:
        print 'Index Error, but continuing...'
        index = np.argsort(norm)[0:(MaxNumPoint/2)] # use the closest points anyway
    
    K  = ( 1 - ( norm[index] / h )**2 ) * 3/4
    # K = np.ones(len(index))

    return index, K

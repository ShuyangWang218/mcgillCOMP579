import numpy as np

class Q_Learning:
    
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes,numberOfBins):
        import numpy as np
        
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.actionNumber=env.action_space.n 
        self.numberEpisodes=numberEpisodes
        self.numberOfBins=numberOfBins
        self.lowerBounds=env.observation_space.low
        self.upperBounds=env.observation_space.high
        
        # this list stores sum of rewards in every learning episode
        self.sumRewardsEpisode=[]
        
        # this matrix is the action value function matrix 
        self.Qmatrix=np.random.uniform(low=0, high=1, size=(numberOfBins,numberOfBins,numberOfBins,numberOfBins,self.actionNumber))


    def discretizatoin(self, state):
        '''This function takes state, the list of exact values of position, velocity, angle and
        angular velocity as input, and returns the indices of the corresponding bins, respectively.
        '''
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]
        
        cartPositionBin=np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numberOfBins)
        cartVelocityBin=np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numberOfBins)
        poleAngleBin=np.linspace(self.lowerBounds[2],self.upperBounds[2],self.numberOfBins)
        poleAngleVelocityBin=np.linspace(self.lowerBounds[3],self.upperBounds[3],self.numberOfBins)
        
        indexPosition=np.maximum(np.digitize(position,cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(velocity,cartVelocityBin)-1,0)
        indexAngle=np.maximum(np.digitize(angle,poleAngleBin)-1,0)
        indexAngularVelocity=np.maximum(np.digitize(angularVelocity,poleAngleVelocityBin)-1,0)
        
        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])   


    def chooseAction(self, state, index):
        '''This function selects an action using epsilon-greedy approach. It takes two parameters,
        state St and the index of the episode.
        '''

        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()

        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qmatrix[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            return np.random.choice(np.where(self.Qmatrix[self.discretizatoin(state)] == np.max(self.Qmatrix[self.discretizatoin(state)]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example 
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple
    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################
    
    
    ###########################################################################
    #    START - function for simulating learning episodes
    ###########################################################################
     
    def simulateEpisodes(self):
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
            
            # reset the environment at the beginning of every episode
            (stateS,_)=self.env.reset()
            stateS=list(stateS)
          
            #print("Simulating episode {}".format(indexEpisode))
            
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            while not terminalState:
                # return a discretized index of the state
                stateSIndex=self.discretizatoin(stateS)
                
                # select an action on the basis of the current state, denoted by stateS
                actionA = self.chooseAction(stateS, indexEpisode)

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (stateSprime, reward, terminalState,_,_) = self.env.step(actionA)          
                
                rewardsEpisode.append(reward)
                
                stateSprime=list(stateSprime)
                
                stateSprimeIndex=self.discretizatoin(stateSprime)
                
                # return the max value, we do not need actionAprime...
                QmaxPrime=np.max(self.Qmatrix[stateSprimeIndex])                                               
                                             
                if not terminalState:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error=reward+self.gamma*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                
                # set the current state to the next state                    
                stateS=stateSprime
        
            #print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))






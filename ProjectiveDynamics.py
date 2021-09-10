import torch
import numpy as np
class pyPD(torch.nn.Module) :

    #refer to [Liu et al. 2017]
    def __init__(self, x, v, S, M, dt) :
        super(ProjectiveDynamics, self).__init__()

        self.x = x                              #initial position
        self.v = v                              #initial velocity
        self.M = M                              #mass matrix 
        self.S = S                              #selection matrix   
        self.e = S.shape[0]//4                  #number of tet elements 
        self.n = x.shape[0]                     #number of tet elements
        self.dt = dt                            #time step 

        Li, Ji_T = self.__compute_L_and_J()     #initialize constant variables

        self.Li = Li 
        self.Ji_T = Ji_T

        A = M + dt * dt * torch.sum(Li)         #Laplacian matrix: M + h*h*L
        self.U = torch.cholesky(A)   
    
    def forward(self) : 
        y = self.x + self.dt * self.v
        xt = y
        itertation = 5

        for i in range(itertation):
            R = self.__compute_R(xt)            #compute projection 
            gf = torch.matmul(self.M, xt - y) + self.dt*self.dt*(torch.sum(torch.matmul(self.Li,xt)) - torch.sum(torch.matmul(self.Ji_T,R)))   # gradient: M(x-y) + h*h*(lx - Jp)
            xt = xt - torch.cholesky_solve(gf,self.U)
            
        self.v = (xt - self.x)/self.dt
        self.x = xt   
        return  self.x

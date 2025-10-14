import time
from math_utils import *
from sph_kernel import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):
        self.ps = particle_system


    def solve(self, dt):

        print("Not implemented yet....")
        pass
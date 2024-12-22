from manim import *
from manim.opengl import *
from scipy import stats
import numpy as np

from utils import *

w=1

class RightAngleIn3D(OpenGLVMobject):
    def __init__(self, line1, line2, length = 0.2, **kwargs):
        # Assumes that line 2 starts where line 1 ends
        
        line1_start = line1.get_start()
        line1_end = line1.get_end()
        line2_start = line2.get_start()
        line2_end = line2.get_end()

        unit1 = (line1_end-line1_start)/np.linalg.norm(line1_end-line1_start)
        start = line1_end - unit1 * length
        unit2 = (line2_end - line2_start)/np.linalg.norm(line2_end - line2_start)
        end = line2_start + unit2 * length

        direct = end - start
        to_intersection = line1_end-start
        
        middle = 2*project(to_intersection,direct)-to_intersection + start                
        middle1 = start + (middle-start)/2
        middle2 = middle + (end - middle)/2
        
        super().__init__()
        self.points=[start,middle1,middle,middle, middle2,end]


config.renderer="opengl"
class test(Scene):
    def construct(self):      
        l1 = Vector(RIGHT)      
        l2 = Vector(UP).shift(RIGHT)
        ra = RightAngleIn3D(l1,l2)
        self.add(l1,l2,ra)


with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()

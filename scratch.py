from manim import *
from scipy import stats
import numpy as np

from utils import *

w=1


class test(Scene):
    def construct(self):
        v = Arrow(ORIGIN,RIGHT,buff=0,color=TEAL)
        n= always_redraw(lambda: Arrow(v.get_start(),v.get_start()+v.get_normal_vector(),buff=0))
        def dot():
            theta, phi, _ = self.camera.euler_angles
            x = np.cos(theta-90*DEGREES) * np.sin(phi)
            y = np.sin(theta-90*DEGREES) * np.sin(phi)
            z = np.cos(phi)
            return Dot([x,y,z])
        d = always_redraw(dot)
        self.add(n,v,d)
        self.interactive_embed()
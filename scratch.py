from manim import *
from manim.opengl import *
from scipy import stats
import numpy as np

from utils import *

w=1



config.renderer="opengl"
class test(Scene):
    def construct(self):      
        dxy = DashedLine(LEFT,RIGHT,dash_length=0.1).set_flat_stroke(True).set_opacity(0.5).scale(4)
        # dxy.submobjects.reverse()
        self.play(Indicate(dxy.family[0][0]))
        dxy.invert()
        self.play(Indicate(dxy.family[0][0]))
        self.play(Write(dxy,reverse=True),run_time=3)
        self.wait()

"""
with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()
"""
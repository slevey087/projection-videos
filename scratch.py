from manim import *
from scipy import stats
import numpy as np

from utils import *

w=1

config.renderer="opengl"
class test(Scene):
    def construct(self):        
        light = SVGMobject("flashlight-svgrepo-com.svg")
        light.data["stroke_width"]=np.array([0])
        light.get_top
        t = MathTex("t")
        self.play(FadeIn(light))

with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()

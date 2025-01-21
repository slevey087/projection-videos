from manim import *
from manim.opengl import *
from scipy import stats
import numpy as np

# from utils import *

w=1


# new colors
COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN



# config.renderer="opengl"
class test(ThreeDScene):
    def construct(self):
        # Set up 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3],
            x_length=6, y_length=6, z_length=6
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.add(axes)

        # Scatter random points
        data_points = [
            np.array([np.random.uniform(-1.5, 1.5),
                      np.random.uniform(-1.5, 1.5),
                      np.random.uniform(-1.5, 1.5)]) for _ in range(60)
        ]
        points = VGroup(*[
            Dot(radius=0.05, color=BLUE).move_to(axes.c2p(*point)).rotate(45*DEGREES,axis=(-1,1,0))
            for point in data_points
        ])
        self.add(points)

        # Principal component vectors
        pc1 = Arrow(start=axes.c2p(0, 0, 0), end=axes.c2p(2, 1, 1),
                    color=RED, stroke_width=4,buff=0)
        pc1.rotate(30*DEGREES,axis=pc1.get_unit_vector())
        pc2 = Arrow(start=axes.c2p(0, 0, 0), end=axes.c2p(-1, 2, 0.5),
                    color=GREEN, stroke_width=4,buff=0)
        pc2.rotate(45*DEGREES,axis=pc2.get_unit_vector())
        pc3 = Arrow(start=axes.c2p(0, 0, 0), end=axes.c2p(0.5, -0.5, 2),
                    color=YELLOW, stroke_width=4,buff=0)
        pc3.rotate(75*DEGREES,axis=pc3.get_unit_vector())

        # Animate
        self.add(pc1, pc2, pc3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)



"""
with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()
"""
from manim import *
from manim.opengl import *
# from scipy import stats
# import numpy as np

from utils import *

w=1





# old color
XCOLOR = XKCD.LIGHTVIOLET
YCOLOR = XKCD.LIGHTYELLOW
VCOLOR = XKCD.AQUA
PCOLOR = XKCD.LIGHTAQUA
RCOLOR = XKCD.LIGHTORANGE

UCOLOR = XKCD.LIGHTBLUE
# VCOLOR = XKCD.LIGHTPURPLE
PUCOLOR = XKCD.LIGHTCYAN
PVCOLOR = XKCD.LIGHTLAVENDER

# new colors
COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN
COLOR_V3P = XKCD.ELECTRICPURPLE # XKCD.EASTERPURPLE and XKCD.LIGHTISHPURPLE are good too

# re-assign colors, to test
XCOLOR = ManimColor.from_hex("#FFC857")
YCOLOR = ManimColor.from_hex("#DB3A34")
PCOLOR = XKCD.WINDOWSBLUE

BCOLOR = XKCD.ELECTRICPURPLE



def color_tex_standard(equation):
    if isinstance(equation,Matrix):
        for entry in equation.get_entries(): color_tex_standard(entry)
        return equation
    return color_tex(equation,(r"\mathbf{v}", VCOLOR), (r"\mathbf{x}",XCOLOR),(r"\mathbf{y}",YCOLOR), (r"\mathbf{p}",PCOLOR),("p_x",PCOLOR),("p_y",PCOLOR), (r"\mathbf{u}", UCOLOR),(r"\hat{\mathbf{u}}", PUCOLOR),(r"\hat{\mathbf{v}}", PVCOLOR))




# config.renderer="opengl"


import numpy as np



class FlatArrow(OpenGLGroup):
    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        stroke_width=0.07,
        tip_length=0.38,
        tip_width=0.42,
        color=WHITE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stroke_width = stroke_width
        self.tip_length = tip_length
        self.tip_width = tip_width if tip_width is not None else 2 * stroke_width
        self.color = color
        self._start = np.array(start)
        self._end = np.array(end)
        self._create_arrow()

    def _create_arrow(self):
        direction = self._end - self._start
        length = np.linalg.norm(direction)
        if length == 0:
            self.submobjects = []
            return

        unit = direction / length
        shaft_length = max(0, length - self.tip_length)

        # Create shaft surface
        def shaft_surface(u, v):
            return np.array([
                u * shaft_length,
                (v - 0.5) * self.stroke_width,
                0
            ])

        shaft = OpenGLSurface(
            shaft_surface,
            u_range=[0, 1],
            v_range=[0, 1],
            # resolution=(2, 2),
            epsilon=0.000001,
            fill_opacity=1,
            checkerboard_colors=[self.color],
        )

        # Create tip surface
        def tip_surface(u, v):
            return np.array([
                v * self.tip_length,
                (0.5 - u) * (1 - v) * self.tip_width,
                0
            ])

        tip = OpenGLSurface(
            tip_surface,
            u_range=[0, 1],
            v_range=[0, 1],
            # resolution=(3, 3),
            epsilon=0.000001,
            fill_opacity=1,
            checkerboard_colors=[self.color],
        )
        tip.shift(RIGHT * shaft_length)  # Position tip at the end of the shaft

        self.submobjects = [shaft, tip]

        # Calculate the center of the combined arrow
        center = self._start + direction / 2

        # Position the combined arrow
        self.move_to(center)

        # Align the arrow with the direction vector
        reference_vector = RIGHT
        if not np.allclose(unit, reference_vector):
            axis = np.cross(reference_vector, unit)
            norm_axis = np.linalg.norm(axis)
            if norm_axis > 1e-6:  # Avoid division by zero for collinear vectors
                angle = np.arccos(np.dot(reference_vector, unit))
                self.rotate(angle, axis=axis / norm_axis, about_point=center)

    def put_start_and_end_on(self, start, end):
        self._start = np.array(start)
        self._end = np.array(end)
        self._create_arrow()
        return self

    def get_vector(self):
        return self._end - self._start

    def get_unit_vector(self):
        vec = self.get_vector()
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else RIGHT

    def scale(self, factor, **kwargs):
        vec = self.get_vector()
        new_end = self._start + factor * vec
        return self.put_start_and_end_on(self._start, new_end)






class test(ThreeDScene):
    def construct(self):
        low_plane_resolution = 16 # increase to like 32 or even 64 for higher quality render (but will take way longer)
        high_plane_resolution = 10
        # Arrow.set_default(shade_in_3d=True)
        
        x1coords = np.array([1,0,0])
        x2coords = np.array([0,1,0])
        b1coords = np.array([1,0,0.35])
        b2coords = np.array([0,1,0.25])
        vcoords = np.array([0.5,0.7,0.7])
        amatrix = np.vstack([x1coords,x2coords]).T
        bmatrix = np.vstack([b1coords,b2coords]).T
        
        px1coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[0]
        px2coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[1]
        pcoords = px1coord*x1coords + px2coord*x2coords
        
        bpx1coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[0]
        bpx2coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[1]
        bpcoords = bpx1coord*b1coords + bpx2coord*b2coords

        # define diagram
        axes = ThreeDAxes(
            x_range=[0,0.5],x_length=2.5 / 2,
            y_range=[0,0.5],y_length=2.5 / 2,
            z_range=[0,0.5],z_length=2.5 / 2,
        ).set_opacity(0)        
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR,shade_in_3d=True).set_stroke(width=6)        
        x1 = Arrow(axes @ ORIGIN, axes @ x1coords, buff=0, color=XCOLOR).set_stroke(width=6)
        x2 = Arrow(axes @ ORIGIN, axes @ x2coords, buff=0, color=XCOLOR).set_stroke(width=6)        
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR).set_stroke(width=6)        
        px1 = Arrow(axes @ ORIGIN, axes @ (px1coord*x1coords), buff=0,color=PCOLOR).set_stroke(width=6)
        px2 = Arrow(axes @ ORIGIN, axes @ (px2coord*x2coords), buff=0,color=PCOLOR).set_stroke(width=6)
        dp = DashedLine(v.get_end(),p.get_end(),dash_length=0.15).set_opacity(0.4)
        dx1 = DashedLine(axes @ pcoords, axes @ (px1coord*x1coords), dash_length=0.15).set_opacity(0.4)
        dx2 = DashedLine(axes @ pcoords, axes @ (px2coord*x2coords), dash_length=0.15).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR).set_stroke(width=6)        
        ArrowGradient(r,[PCOLOR,VCOLOR])

        b1 = Arrow(axes @ ORIGIN, axes @ b1coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)
        b2 = Arrow(axes @ ORIGIN, axes @ b2coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)

        angle = Arc3d(p.get_center(),r.get_center(),p.get_end(),radius=0.4).set_stroke(opacity=0.4)
        vectors = VGroup(v,x1,x2,p,px1,px2,r).set_shade_in_3d()
        b_vectors = VGroup(b1,b2).set_shade_in_3d()
        dashes = VGroup(dp,dx1,dx2)

        plane =  Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=low_plane_resolution).set_stroke(width=0.06,opacity=0.5).set_opacity(0.5).set_color(ManimColor('#29ABCA'))
        
        reference_dot=Dot().move_to(axes @ ORIGIN).set_opacity(0)
        for mob in [*plane,*x1,*x2,*px1,*px2,*p,*b1,*b2]: mob.z_index_group=reference_dot
        
        
        plane2 = Surface(lambda u,v:axes @ (u*b1coords+v*b2coords),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.039,resolution=low_plane_resolution).set_opacity(0.5).set_color(BCOLOR)
        

        diagram = Group(reference_dot,axes,plane,vectors,dashes, angle,plane2,b_vectors)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        x1l = MathTex(r"\mathbf{x_1}", color=XCOLOR, font_size=50).next_to(x1.get_end(),DOWN,buff=0.1)
        x2l = MathTex(r"\mathbf{x_2}", color=XCOLOR, font_size=50).next_to(x2.get_end(),DR,buff=0.1)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        px1l = MathTex(r"p_1 \mathbf{x_1}", font_size=40).next_to(px1.get_end(),LEFT,buff=0.15).shift(UP*0.15)
        color_tex_standard(px1l)        
        px2l = MathTex(r"p_2 \mathbf{x_2}", font_size=40).next_to(px2.get_end(),UP,buff=0.15)
        color_tex_standard(px2l)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.5)
        color_tex_standard(rl)
        b1l = MathTex(r"\mathbf{b_1}",font_size=40,color=BCOLOR.lighter()).next_to(b1.get_end(),LEFT)
        b2l = MathTex(r"\mathbf{b_2}",font_size=50,color=BCOLOR.lighter()).next_to(b2.get_end(),RIGHT)
        labels = VGroup(x1l,vl,x2l,pl,px1l,px2l,rl,b1l,b2l)
        diagram.add(labels) 
  
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.35+RIGHT*0.2)
        self.set_camera_orientation(frame_center=IN*11) # self.set_camera_orientation(zoom=2)
        for vector in vectors+[b_vectors]: 
            ArrowStrokeFor3dScene(self,vector,family=True)
        face_camera(self,r)
        ArrowGradient(r,[PCOLOR,VCOLOR])

        self.add(diagram)
        
        


# config.from_animation_number = 2
# config.upto_animation_number = 3





with tempconfig({"quality": "medium_quality"}):
    scene = test()
    scene.render()

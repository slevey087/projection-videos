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



class test1(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(zoom=1.5)

        plane = Surface(lambda u,v: (u,v,0),u_range=[-2,2],v_range=[-2,2],resolution=8).set_opacity(0.6)
        v1 = Arrow(2*LEFT,2*RIGHT,buff=0, color=YELLOW)
        v2 = Arrow(2*LEFT,RIGHT+UP,buff=0,color=RED)

        plane2 = Surface(lambda u,v: (u,v,0),u_range=[-2,2],v_range=[-2,2],resolution=8,color=GREEN).set_opacity(0.6).shift(IN*0.5)
        
        diagram = VGroup(plane,v1,v2).set_shade_in_3d(True,True)
        diagram.add(plane2)
        self.add(diagram)

        self.wait()
        self.play(diagram.animate.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT),run_time=2) 
        self.wait()
        plane2.add_updater(lambda p: p.set_shade_in_3d(True))
        self.play(plane2.animate.shift(2*(plane.get_center()-plane2.get_center())),run_time=2)
        self.wait()


class test(ThreeDScene):
    def construct(self):
        Arrow.set_default(shade_in_3d=True)
        
        xcoords = np.array([1,0,0])
        ycoords = np.array([0,1,0])
        b1coords = normalize(np.array([1,0,0.35]))
        b2coords = normalize(np.array([0,1,0.15]))
        vcoords = np.array([0.5,0.7,0.7])
        amatrix = np.vstack([xcoords,ycoords]).T
        bmatrix = np.vstack([b1coords,b2coords]).T
        
        pxcoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[0]
        pycoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[1]
        pcoords = pxcoord*xcoords + pycoord*ycoords
        
        bpxcoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[0]
        bpycoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[1]
        bpcoords = bpxcoord*b1coords + bpycoord*b2coords

        # define diagram
        axes = ThreeDAxes(
            x_range=[0,0.5],x_length=2.5 / 2,
            y_range=[0,0.5],y_length=2.5 / 2,
            z_range=[0,0.5],z_length=2.5 / 2,
        ).set_opacity(0)        
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR,shade_in_3d=True).set_stroke(width=6)        
        x = Arrow(axes @ ORIGIN, axes @ xcoords, buff=0, color=XCOLOR).set_stroke(width=6)
        y = Arrow(axes @ ORIGIN, axes @ ycoords, buff=0, color=YCOLOR).set_stroke(width=6)        
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR).set_stroke(width=6)        
        px = Arrow(axes @ ORIGIN, axes @ (pxcoord*xcoords), buff=0,color=PCOLOR).set_stroke(width=6)
        py = Arrow(axes @ ORIGIN, axes @ (pycoord*ycoords), buff=0,color=PCOLOR).set_stroke(width=6)
        dp = DashedLine(v.get_end(),p.get_end(),dash_length=0.15).set_opacity(0.4)
        dy = DashedLine(axes @ pcoords, axes @ (pxcoord*xcoords), dash_length=0.15).set_opacity(0.4)
        dx = DashedLine(axes @ pcoords, axes @ (pycoord*ycoords), dash_length=0.15).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR).set_stroke(width=6)        
        ArrowGradient(r,[PCOLOR,VCOLOR])

        b = Arrow(axes @ ORIGIN, axes @ b1coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)
        c = Arrow(axes @ ORIGIN, axes @ b2coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)

        angle = Arc3d(p.get_center(),r.get_center(),p.get_end(),radius=0.4).set_stroke(opacity=0.4)
        vectors = VGroup(v,x,y,p,px,py,r)
        b_vectors = VGroup(b,c)
        dashes = VGroup(dp,dy,dx)

        plane =  Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.1,resolution=(64,16)).set_opacity(0.4).set_color(ManimColor('#29ABCA'))
        for mob in [*plane,*x,*y,*px,*py,*p,*b,*c]: mob.z_index_group=Dot()
        
        
        plane2 = Surface(lambda u,v:axes @ (u*b1coords+v*b2coords),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.1,resolution=32).set_opacity(0.4).set_color(BCOLOR)
        VGroup(plane,plane2).set_stroke(opacity=1)

        diagram = Group(axes,plane,vectors,dashes, angle,plane2,b_vectors)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UL,buff=0.15)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UP,buff=0.15)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl)
        diagram.add(labels)                
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.35+RIGHT*0.2)
        self.set_camera_orientation(frame_center=IN*11) # self.set_camera_orientation(zoom=2)
        for vector in vectors+[b_vectors]: 
            ArrowStrokeFor3dScene(self,vector,family=True)
        face_camera(self,r)
        ArrowGradient(r,[PCOLOR,VCOLOR])

        self.add(diagram)
        diagram.rotate(40*DEGREES, axis=(axes @ (vcoords - pcoords))-(axes @ ORIGIN), about_point=diagram.get_center())
        diagram.rotate(10*DEGREES,axis=(axes @ b2coords) - (axes @ ORIGIN),about_point=diagram.get_center())
        self.remove(x,xl,px,pxl,p,pl,r,rl,y,yl,py,pyl,dx,dy,dp,plane,angle)
        self.interactive_embed()


# config.from_animation_number = 2
# config.upto_animation_number = 3





with tempconfig({"quality": "medium_quality", "dry_run":True}):
    scene = test()
    scene.render()

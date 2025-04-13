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




config.renderer="opengl"

class test(ThreeDScene):
    def construct(self):
        Arrow.set_default(flat_stroke=False,shade_in_3d=True)


        xcoords = np.array([1,0,0])
        ycoords = np.array([0,1,0])
        b1coords = np.array([1,0,0.35])
        b2coords = np.array([0,1,0.15])
        vcoords = np.array([0.5,0.7,0.7])
        amatrix = np.vstack([xcoords,ycoords]).T
        bmatrix = np.vstack([b1coords,b2coords]).T
        
        pxcoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[0]
        pycoord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[1]
        pcoords = pxcoord*xcoords + pycoord*ycoords
        

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
        # ArrowGradient(r,[PCOLOR,VCOLOR])

        angle = Arc3d(p.get_center(),r.get_center(),p.get_end(),radius=0.4).set_stroke(opacity=0.4)
        vectors = VGroup(v,x,y,p,px,py,r)
        dashes = VGroup(dp,dy,dx)

        plane =  Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.1,resolution=10).set_opacity(0.6).set_color(ManimColor('#29ABCA'))
        plane2 = Surface(lambda u,v:axes @ (u*b1coords+v*b2coords),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.1,resolution=10).set_opacity(0.6).set_color(BCOLOR)
        
        diagram = Group(axes,plane,plane2,vectors,dashes, angle)
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
        """
        self.set_camera_orientation(frame_center=IN*11) # self.set_camera_orientation(zoom=2)
        for vector in vectors: 
            ArrowStrokeFor3dScene(self,vector,family=True)
        face_camera(self,r)
        ArrowGradient(r,[PCOLOR,VCOLOR])
        """
        self.camera.scale(0.5)

        self.add(diagram)
        diagram.rotate(40*DEGREES, axis=(axes @ (vcoords - pcoords))-(axes @ ORIGIN), about_point=diagram.get_center())
        diagram.rotate(10*DEGREES,axis=(axes @ b2coords) - (axes @ ORIGIN),about_point=diagram.get_center())
        self.interactive_embed()


# config.from_animation_number = 2
# config.upto_animation_number = 3





with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()

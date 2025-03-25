from manim import *
from scipy import stats
import numpy as np

from utils import *


from sklearn.linear_model import LinearRegression

w=1



COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN
COLOR_V3P = XKCD.ELECTRICPURPLE # XKCD.EASTERPURPLE and XKCD.LIGHTISHPURPLE are good too


XCOLOR = ManimColor.from_hex("#FFC857").darker()
YCOLOR = ManimColor.from_hex("#DB3A34").darker()
VCOLOR = XKCD.AQUA
PCOLOR = XKCD.WINDOWSBLUE
RCOLOR = XKCD.LIGHTORANGE

UCOLOR = XKCD.LIGHTBLUE
# VCOLOR = XKCD.LIGHTPURPLE
PUCOLOR = XKCD.LIGHTCYAN
PVCOLOR = XKCD.LIGHTLAVENDER





UNDERLINE_COLOR = XKCD.LIGHTBLUE



def color_tex_standard(equation):
    if isinstance(equation,Matrix):
        for entry in equation.get_entries(): color_tex_standard(entry)
        return equation
    return color_tex(equation,
        (r"\mathbf{v}", VCOLOR), 
        (r"\mathbf{x}",XCOLOR),
        (r"\mathbf{y}",YCOLOR), 
        (r"\mathbf{p}",PCOLOR),
        ("p_x",PCOLOR),("p_y",PCOLOR), 
        (r"\mathbf{u}", UCOLOR),
        (r"\hat{\mathbf{u}}", PUCOLOR),
        (r"\hat{\mathbf{v}}", PVCOLOR))




class Oblique1D(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,0.7])
        vcoords = np.array([1,2.2])
        k = 0.55 # parameter to control degree of oblique projection. At 1, it's the orthogonal projection; at 0, it's 0.
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * k
        zcoords = np.array([1,-(vcoords - pcoords)[0] / ((vcoords - pcoords)[1])]) * 2 # formula here is based on tha the dot product with r must be 0
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[0,2], x_length=2,y_range=[0,2],y_length=2).set_opacity(0)

        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        p = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0)                
        ArrowGradient(r,[PCOLOR,VCOLOR])
        z = Arrow(axes @ ORIGIN, axes @ zcoords,buff=0, color=COLOR_V3P)
        vectors = VGroup(x,v,p,r,z)       

        angle = Angle(p,r,radius=0.35,quadrant=(-1,1),other_angle=True) 
        dx = DashedLine(v.get_end(),p.get_end(),dash_length=0.1).set_opacity(0.6)

        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DR,buff=0.03)        
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r.get_center()).shift(UP*0.1+RIGHT*0.05)
        color_tex_standard(rl)
        zl = MathTex(r"\mathbf{z}", font_size=60, color=COLOR_V3P).next_to(z.get_tip(),buff=0.15)        
        labels = VGroup(xl, vl, pl, rl, zl)
        
        diagram = VGroup(axes, vectors, labels,angle,dx)
        diagram.to_corner(UR)
        
        frame.move_to(diagram)

        # start drawing diagram
        self.play(GrowArrow(x))
        self.play(GrowArrow(v))       
        self.play(Write(xl))
        self.play(Write(vl))
        self.wait(w)

        # zoom in        
        self.play(frame.animate.scale(0.5).move_to(VGroup(x,v).get_center()), run_time=2)
        self.wait()

        # draw p and label. Also follow with dash then remove dash     
        self.play(
            TransformFromCopy(v,p), 
            Write(dx),
            run_time=2
        )       
        self.play(
            Write(pl),
            FadeOut(dx),
            run_time=2
        )
        self.wait(w)

        # projection
        proj = Tex("Projection").next_to(pl,DOWN,aligned_edge=UP,buff=0.3)        
        self.play(DrawBorderThenFill(proj))
        self.wait(w)
        self.play(DrawBorderThenFill(proj, reverse_rate_function=True))
        self.wait(w)

        # equation for p
        pe = MathTex(r"\mathbf{p}","=",r"p_x \mathbf{x}", font_size=60)        
        color_tex_standard(pe)
        pe.shift(pl[0].get_center()-pe[0].get_center())
        self.play(
            ReplacementTransform(pl[0],pe[0]),
            Write(pe[1])
        )
        self.play(Write(pe[2]))
        self.wait(w)

        # remove p equation
        self.play(FadeOut(pe[1:]))
        self.wait(w)

        # rejection and label        
        self.play(GrowArrow(r))        
        self.play(Write(rl))
        self.wait(w)

        # rejection text
        rej = Tex("Rejection").next_to(rl,UP).shift(DOWN*0.15)#.align_to(rl,LEFT)
        self.play(DrawBorderThenFill(rej))
        self.wait(w)
        self.play(DrawBorderThenFill(rej, reverse_rate_function=True))
        self.wait(w)

        # angle        
        self.play(ReplacementTransform(angle.copy().scale(20).shift(LEFT*3).set_opacity(0),angle),run_time=2)
        self.wait(w)

        # animation where angle moves
        self.remove(r,rl,p,angle)
        k_anim = ValueTracker(k)
        p_anim = always_redraw(lambda: Arrow(axes @ ORIGIN, axes @ ((xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords))*k_anim.get_value()), buff=0, color=PCOLOR))
        r_anim = always_redraw(lambda: ArrowGradient(Arrow(p_anim.get_end(), v.get_end(), buff=0),[PCOLOR,VCOLOR]))        
        rl_anim = always_redraw(lambda: color_tex_standard(MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r_anim.get_center()).shift(UP*0.1+RIGHT*0.05)))
        angle_anim = always_redraw(lambda: Angle(p_anim,r_anim,radius=0.35,quadrant=(-1,1),other_angle=True if k_anim.get_value()>=0 else False) )
        self.add(p_anim,r_anim, rl_anim,angle_anim)
        self.play(k_anim.animate.set_value(1.6),run_time=2)
        self.play(k_anim.animate.set_value(-0.5),run_time=2)
        self.play(k_anim.animate.set_value(k),run_time=2)
        self.remove(p_anim,r_anim, rl_anim,angle_anim)
        self.add(r,rl,p,angle)
        self.wait()

        # add z vector
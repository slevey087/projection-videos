from manim import *
from scipy import stats
import numpy as np

from utils import *

w=1

XCOLOR = XKCD.LIGHTVIOLET
YCOLOR = XKCD.LIGHTYELLOW
VCOLOR = XKCD.AQUA
PCOLOR = XKCD.LIGHTAQUA
RCOLOR = XKCD.LIGHTORANGE

UCOLOR = XKCD.LIGHTBLUE
# VCOLOR = XKCD.LIGHTPURPLE
PUCOLOR = XKCD.LIGHTCYAN
PVCOLOR = XKCD.LIGHTLAVENDER


def color_tex_standard(equation):
    if isinstance(equation,Matrix):
        for entry in equation.get_entries(): color_tex_standard(entry)
        return equation
    return color_tex(equation,(r"\mathbf{v}", VCOLOR), (r"\mathbf{x}",XCOLOR),(r"\mathbf{y}",YCOLOR), (r"\mathbf{p}",PCOLOR),("p_x",PCOLOR),("p_y",PCOLOR), (r"\mathbf{u}", UCOLOR),(r"\hat{\mathbf{u}}", PUCOLOR),(r"\hat{\mathbf{v}}", PVCOLOR))


class Introduction(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,1])
        vcoords = np.array([0.5,2])
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        vhat = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)
        vhatl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=PCOLOR).next_to(vhat.get_tip(), DOWN,buff=0.1)
        r = DashedLine(axes.c2p(*vcoords),axes.c2p(*pcoords), dash_length=0.12).set_opacity(0.5)
        ra = RightAngle(vhat,r,length=0.25,quadrant=(-1,-1))
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60, color=RCOLOR).next_to(r).shift(UP*0.2+LEFT*0.3)
        diagram = VGroup(axes,x,v,xl,vl,r,vhat,vhatl,ra).shift(-VGroup(v,x).get_center()).shift(LEFT*2.25)
        diagram.remove(axes)        

        frame.scale(0.5).move_to(VGroup(x,v))

        # draw x and v
        self.play(GrowArrow(x))
        self.play(Write(xl))
        self.play(GrowArrow(v))
        self.play(Write(vl))        
        self.wait(w)

        # drop projection
        self.play(
            TransformFromCopy(v,vhat),
            Write(r)
        ,run_time=2)
        self.play(Write(vhatl))        
        self.wait(w)

        # write title
        title = Tex("Orthogonal Projection", font_size=50).next_to(diagram,UP,buff=0.5)
        ul = Line(title.get_corner(DL)+DL*0.1, title.get_corner(DR)+DR*0.1, color=BLUE)        
        self.play(Write(ra))
        self.play(frame.animate.scale(1.3).shift(UP*0.3),run_time=1.5)
        self.play(Write(title))
        self.play(Write(ul))
        self.wait(w)

        # shift diagram, write equation
        eq = MathTex(r"\hat{\mathbf{v}}","=","P",r"\mathbf{v}",font_size=70).move_to(diagram).shift(RIGHT*1.5).shift(UP*0.5)
        color_tex_standard(eq)
        self.play(diagram.animate.shift(LEFT*1.5))
        self.play(TransformFromCopy(vl[0],eq[3]),run_time=1.5)
        self.play(FadeIn(eq[2],shift=RIGHT),run_time=1.5)
        self.play(Write(eq[1]),run_time=1.25)
        self.play(TransformFromCopy(vhatl[0],eq[0]),run_time=1.25)
        
        self.wait(w)
        
        


class Idempotent(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,1])
        vcoords = np.array([0.5,2])
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0, color=RCOLOR)        
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60, color=RCOLOR).next_to(r).shift(UP*0.2+LEFT*0.3)
        diagram = VGroup(axes,x,v,xl,vl).shift(-VGroup(v,x).get_center()).shift(LEFT*2.25)
        diagram.remove(axes)
        frame.scale(0.5).move_to(VGroup(x,v))
        
        # fade in diagram
        self.play(FadeIn(diagram),run_time=2)
        self.wait(w)

        # project
        p = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)
        self.play(TransformFromCopy(v,p), run_time=2)
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DOWN)        
        diagram.add(p,pl)
        self.play(Write(pl))
        self.wait(w)

        # p equation
        peq = MathTex(r"\mathbf{p}","=","P",r"\mathbf{v}",font_size=70)
        peq.move_to([-diagram.get_center()[0],diagram.get_center()[1],0])
        AlignBaseline(peq, xl)
        color_tex_standard(peq)
        self.play(frame.animate.move_to(VGroup(diagram,peq).get_center()).scale(1.2), run_time=2)
        self.play(TransformFromCopy(vl,peq[3]),run_time=2)                
        self.play(TransformFromCopy(pl, peq[0]), run_time=1.5)
        self.play(Write(peq[1]))        
        self.play(Write(peq[2]))
        self.wait()

        # caption
        p2 = Tex("Projecting Twice ","=", " Projecting Once").align_to(frame,UP).shift(DOWN*0.25)
        self.play(Write(p2))
        self.play(Wiggle(p,scale_value=1.25,rotation_angle=0.05*TAU,run_time=2,n_wiggles=10), Wiggle(pl,scale_value=1.25,rotation_angle=0.05*TAU,run_time=2,n_wiggles=10))
        self.wait(w)

        # p2 = p
        p2ep = MathTex("P^2","=","P", font_size=80).next_to(peq,UP)
        self.play(
            TransformFromCopy(p2[2],p2ep[2]),
            peq.animate.shift(DOWN*0.5)
        ,run_time=1.25)
        self.play(TransformFromCopy(p2[1],p2ep[1]),run_time=1.25)
        self.play(TransformFromCopy(p2[0],p2ep[0]),run_time=1.25)
        self.wait(w)

        # idempotent caption
        title = Tex("Idempotent", font_size=50).move_to(p2)
        ul = Line(title.get_corner(DL)+DL*0.1, title.get_corner(DR)+DR*0.1, color=BLUE)        
        self.play(ReplacementTransform(p2, title))
        self.play(Write(ul))
        self.wait()

        # zoom out and fade out diagram
        self.play(
            FadeOut(*self.mobjects),
            Restore(frame)
        )

        
class SymmetricIntro(Scene):
    def construct(self):
        # title
        title = Tex("Symmetric", font_size=83).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.1, title.get_corner(DR)+DR*0.1, color=BLUE)        
        self.play(Write(title))
        self.play(Write(ul))
        self.wait(w)

        # ptp
        sym = MathTex("P","=","P", font_size=80)
        sym1 = MathTex("P^T","=","P", font_size=80)
        AlignBaseline(sym1,sym)
        self.play(Write(sym))
        self.play(
            ReplacementTransform(sym[0][0],sym1[0][0]), # P
            ReplacementTransform(sym[1],sym1[1]), # =
            ReplacementTransform(sym[2],sym1[2]),
            FadeIn(sym1[0][1],shift=DL)
        )
        self.wait(w)

        # to black
        self.play(FadeOut(*self.mobjects))


# config.renderer = "opengl"
class SymmetricGeom2d(ThreeDScene):
    def construct(self):        
        h = 0.6
        ucoords = np.array([0.7,0.25,h])
        pucoords = ucoords - np.array([0,0,h])        
        vcoords = np.array([0.25,0.7,h])
        pvcoords = vcoords - np.array([0,0,h])

        # set up frame
        frame = self.camera
        original_frame = frame.copy()

        # define diagram
        axes = ThreeDAxes(
            x_range=[-2,2],x_length=10,
            y_range=[-2,2],y_length=10,
            z_range=[-2,2],z_length=10,
        )                
        u = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=UCOLOR)        
        v = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*vcoords), buff=0, color=VCOLOR)                
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=PUCOLOR)        
        pv = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pvcoords), buff=0, color=PVCOLOR)        
        plane = OpenGLSurface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5])
        diagram = Group(plane,u,v,pu,pv)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)
        diagram.shift(LEFT*2.25)
        ul = MathTex(r"\mathbf{u}", color=UCOLOR, font_size=50).next_to(u.get_end(),buff=0.15)
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)        
        pul = MathTex(r"\hat{\mathbf{u}}", color=PUCOLOR, font_size=50).next_to(pu.get_end(),DR,buff=0.05)
        pvl = MathTex(r"\hat{\mathbf{v}}", color=PVCOLOR, font_size=50).next_to(pv.get_end(),DR,buff=0.05)        
        diagram.add(ul,vl,pul,pvl)        
        frame.scale(0.425).move_to(diagram).shift(DOWN*0.25)
        frame.save_state()
        frame.move_to(Group(u,v)).set_euler_angles(phi=-47*DEGREES,theta=11*DEGREES)

        # add vectors
        self.play(ReplacementTransform(u.copy().scale(0.05,u.points[0]),u))        
        self.play(Write(ul))
        self.play(ReplacementTransform(v.copy().scale(0.05,v.points[0]),v))
        self.play(Write(vl))
        self.wait(w)           

        # add in plane
        self.play(
            FadeIn(plane),
            Restore(frame)
        ,run_time=2)
        self.wait(w)

        # project
        self.play(TransformFromCopy(u,pu),run_time=1.25)        
        self.play(Write(pul))
        self.play(TransformFromCopy(v,pv),run_time=1.25)
        self.play(Write(pvl))
        self.wait(w)  

        # dot product
        dot1 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}", font_size=45).next_to(plane,DOWN).shift(LEFT*2+DOWN*0.1)
        color_tex_standard(dot1)
        self.play(TransformFromCopy(pul[0],dot1[0]),run_time=1.25)
        self.play(FadeIn(dot1[1],shift=DOWN),run_time=1.25)
        self.play(TransformFromCopy(pvl[0],dot1[2]),run_time=1.5)
        self.wait(w)

        # project u again, dim vbar
        self.play(
            Group(pv,pvl).animate.set_opacity(0.2),            
            Group(u,ul).animate.set_opacity(0.2),            
        )
        # self.play(Merge([pu,u.copy()],pu),run_time=1.5)
        dot2 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}", font_size=45).move_to(dot1,aligned_edge=LEFT)
        color_tex_standard(dot2)
        AlignBaseline(dot2,dot1)
        self.play(
            ReplacementTransform(dot1[:3],dot2[:3]), # first dot product
            Write(dot2[3]), # =
        )
        self.play(TransformFromCopy(pul[0],dot2[4])) # uhat
        self.play(Write(dot2[5])) # dot
        self.play(TransformFromCopy(vl[0],dot2[6]),run_time=1.75)
        self.wait(w)

        # to other pair
        self.play(
            Group(pv,pvl).animate.set_opacity(1),            
            Group(u,ul).animate.set_opacity(1),            
            Group(v,vl).animate.set_opacity(0.2),            
            Group(pu,pul).animate.set_opacity(0.2),            
        )
        # self.play(Merge([pv,v.copy()],pv),run_time=1.5)
        dot3 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}","=",r"\mathbf{u}","\cdot",r"\hat{\mathbf{v}}", font_size=45).move_to(dot2,aligned_edge=LEFT)
        color_tex_standard(dot3)
        AlignBaseline(dot3,dot2)
        self.play(
            ReplacementTransform(dot2[:7],dot3[:7]), # first two dot products
            Write(dot3[7]), # =
        )
        self.play(TransformFromCopy(ul[0],dot3[8]),run_time=1.75) # u
        self.play(Write(dot3[9])) # dot
        self.play(TransformFromCopy(pvl[0],dot3[10]),run_time=1.25)
        self.wait(w)


        

class SymmetricGeom1d(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,1])
        ucoords = np.array([-2,1])
        vcoords = np.array([0.5,2])                
        pucoords = xcoords * np.dot(xcoords,ucoords) / np.dot(xcoords,xcoords)
        pvcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4).rotate(-15*DEGREES)
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        u = Arrow(axes.c2p(0,0), axes.c2p(*ucoords), buff=0, color=UCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        pu = Arrow(axes.c2p(0,0), axes.c2p(*pucoords), buff=0, color=PUCOLOR)
        pv = Arrow(axes.c2p(0,0), axes.c2p(*pvcoords), buff=0, color=PVCOLOR)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        ul = MathTex(r"\mathbf{u}", font_size=60, color=UCOLOR).next_to(u.get_tip(), UP)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        pul = MathTex(r"\hat{\mathbf{u}}", font_size=60, color=PUCOLOR).next_to(pu.get_tip(), LEFT)
        pvl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=PVCOLOR).next_to(pv.get_tip(), DOWN,buff=0.15)        
        diagram = VGroup(axes,x,u,v,pu,pv,xl,ul,vl,pul,pvl).shift(-VGroup(u,v,x).get_center()).shift(UP)
        diagram.remove(axes)
        frame.scale(0.5).move_to(VGroup(x,u,v))
        
        # fade in diagram
        for vector, label in zip([x,v,u],[xl,vl,ul]):
            self.play(GrowArrow(vector))
            self.play(Write(label))
        self.wait(w)

        # project vectors
        dash_u = DashedLine(u.get_end(),pu.get_end(),dash_length=0.1).set_opacity(0.5)
        dash_v = DashedLine(v.get_end(),pv.get_end(),dash_length=0.1).set_opacity(0.5)
        for vector,original, label,orig_label,dash in zip([pu,pv],[u,v],[pul,pvl],[ul,vl],[dash_u,dash_v]):
            self.play(
                TransformFromCopy(original, vector),
                TransformFromCopy(orig_label,label),
                Write(dash)
            ,run_time=2)                   
        self.wait(w)                

        # dot product
        dot1 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}", font_size=65).next_to(diagram,DOWN,buff=0.3)
        color_tex_standard(dot1)
        self.play(
            frame.animate.scale(1.2).move_to(VGroup(diagram,dot1)),
            FadeOut(dash_u,dash_v)
        )
        self.play(TransformFromCopy(pul[0],dot1[0]),run_time=1.25)
        self.play(FadeIn(dot1[1],shift=DOWN),run_time=1.25)
        self.play(TransformFromCopy(pvl[0],dot1[2]),run_time=1.5)
        self.wait(w)

        # dim vbar                
        dot2 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}", font_size=65).move_to(dot1)
        color_tex_standard(dot2)
        AlignBaseline(dot2,dot1)
        self.play(ReplacementTransform(dot1[:3],dot2[:3])) # first dot product            
        self.play(Write(dot2[3])) # =
        self.play(
            VGroup(pv,pvl).animate.set_opacity(0.15),            
            VGroup(u,ul).animate.set_opacity(0.15),            
            VGroup(x,xl).animate.set_opacity(0.15),            
        )
        self.play(TransformFromCopy(pul[0],dot2[4]),run_time=1.5) # uhat
        self.play(Write(dot2[5])) # dot
        self.play(TransformFromCopy(vl[0],dot2[6]),run_time=1.75)
        self.wait(w)

        # to other pair                
        dot3 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}","=",r"\mathbf{u}","\cdot",r"\hat{\mathbf{v}}", font_size=65).move_to(dot2)
        color_tex_standard(dot3)
        AlignBaseline(dot3,dot2)
        self.play(ReplacementTransform(dot2[:7],dot3[:7])) # first two dot products            
        self.play(Write(dot3[7])) # =
        self.play(
            VGroup(pv,pvl).animate.set_opacity(1),            
            VGroup(u,ul).animate.set_opacity(1),            
            VGroup(v,vl).animate.set_opacity(0.15),            
            VGroup(pu,pul).animate.set_opacity(0.15),            
        )
        self.play(TransformFromCopy(ul[0],dot3[8]),run_time=1.75) # u
        self.play(Write(dot3[9])) # dot
        self.play(TransformFromCopy(pvl[0],dot3[10]),run_time=1.25)
        self.wait(w)







class Symmetric(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Orthogonal projection matrices are symmetric", font_size=55)
        self.play(Write(text1))
        self.wait()
        self.play(text1.animate.next_to(ul, DOWN))
        self.wait()

        # p=pt
        text2 = MathTex("P^T=P", font_size=65)
        self.play(Write(text2))
        self.wait()

        # clear out
        self.play(FadeOut(*self.mobjects))
        self.wait()

        # u,v 
        u = MathTex("u", font_size=65).move_to(UP*3+LEFT*3)
        v = MathTex("v", font_size=65).move_to(UP*3+RIGHT*3)
        self.play(Write(u))
        self.play(Write(v))
        self.wait()

        # extend
        ue = MathTex(r"u=u_{\parallel}+u_{\perp}", font_size=65).move_to(UP*3+LEFT*3)
        ve = MathTex(r"v=v_{\parallel}+v_{\perp}", font_size=65).move_to(UP*3+RIGHT*3)
        self.play(
            ReplacementTransform(u, ue),
            ReplacementTransform(v, ve)
        )
        self.wait()

        # indicate
        self.play(
            Indicate(ue[0][slice(2,4)]),
            Indicate(ve[0][slice(2,4)])
        )
        self.wait()
        self.play(
            Indicate(ue[0][slice(5,None)]),
            Indicate(ve[0][slice(5,None)])
        )
        self.wait()

        # projection
        proju = MathTex("Pu=u_{\parallel}", font_size=65).next_to(ue, DOWN)
        projv = MathTex("Pv=v_{\parallel}", font_size=65).next_to(ve, DOWN)
        self.play(
            Write(proju),
            Write(projv)
        )
        self.wait()

        # Pu dot v
        pudotv = MathTex(r"Pu \cdot v", font_size=65).next_to(proju, DOWN*4)        
        self.play(Write(pudotv))
        self.wait()
        

        # substitute into pu dot v
        eql = AlignBaseline(MathTex("=", font_size=55).next_to(pudotv, RIGHT), pudotv)
        self.play(Write(eql))
        pudotvs = MathTex(r"u_{\parallel}\cdot \left(v_{\parallel}+v_{\perp} \right)", font_size=65).next_to(eql, RIGHT)
        AlignBaseline(pudotvs, pudotv)        
        anims = TransformBuilder(
            [pudotv, ve, proju], pudotvs,
            [
                ([2, 0, slice(3,None)], [0, slice(0,2)], TransformFromCopy), #u_para
                ([0,0,slice(0,2)], None, Indicate), (2, None, Indicate),  # indicate 
                ([0,0,2], [0, 2], TransformFromCopy), # dot
                (None, [0, 3]), (None, [0,-1]), # parentheses
                ([1, 0, slice(2, None)], [0, slice(-6,-1)], TransformFromCopy), # v components
                ([0,0,3], None, Indicate), (1, None, Indicate)   # indicate 
            ]
        )
        self.play(*anims[0:3])
        self.play(*anims[3:6])
        self.play(*anims[6:])
        self.wait()
        
        # distribute dots
        udistr = MathTex(r"u_{\parallel} \cdot v_{\parallel} + u_{\parallel} \cdot v_{\perp}", font_size=65).next_to(eql)
        AlignBaseline(udistr, pudotvs)
        self.play(*TransformBuilder(
            pudotvs, udistr,
            [
                ([0, slice(0,3)], [0, slice(0,3)]), # upara1 dot
                ([0,3], None), ([0,-1], None), #parentheses
                ([0,slice(4,6)], [0, slice(3,5)]), # vpara
                ([0, 6], [0,5]), #plus                
                ([0,slice(0,2)], [0,slice(6,8)], TransformFromCopy, {"path_arc":TAU/4}), #upara2
                ([0, 2], [0, 8], TransformFromCopy,{"path_arc":TAU/4}), # dot 2
                ([0, slice(7,9)], [0, slice(-2,None)]), # vperp
            ]
        ))
        self.wait()
        
        # terms to zero
        zerol = MathTex("0", font_size=65).move_to(udistr[0][8])
        AlignBaseline(zerol, udistr)        
        self.play(
            FadeOut(udistr[0][6:], shift=DOWN),
            FadeIn(zerol, shift=DOWN)
        )
        self.wait()

        # clean up
        puve = AlignBaseline(MathTex(r"Pu\cdot v = u_{\parallel} \cdot v_{\parallel}", font_size=65).align_to(pudotv, LEFT).shift(LEFT*2.5), pudotv)        
        self.play(
            FadeOut(zerol, udistr[0][5]),
            TransformMatchingShapes(VGroup(pudotv, udistr[0][:5], eql), puve)
        )
        self.wait()
        

        # u dot pv
        udotpv = MathTex(r"u \cdot Pv", font_size=65).next_to(projv, DOWN*4)
        self.play(Write(udotpv))
        self.wait()

        # substitute into u dot pv
        eqr = AlignBaseline(MathTex("=", font_size=55).next_to(udotpv).shift(LEFT*2.5), udotpv)
        self.play(
            Write(eqr),
            udotpv.animate.shift(LEFT*2.5)
        )
        udotpvs = MathTex(r"\left( u_{\parallel} + u_{\perp} \right) \cdot v_{\parallel}", font_size=65).next_to(eqr)
        AlignBaseline(udotpvs, udotpv)
        self.play(*TransformBuilder(
            [udotpv, ue, projv], udotpvs,
            [
                ([1,0, slice(-5,None)], [0,slice(1,6)], TransformFromCopy), # u components
                (None, [0, 0]), (None, [0,6]), #parentheses
                ([0,0,1], [0,7], TransformFromCopy), #dot
                ([2,0, slice(-2,None)], [0,slice(-2,None)], TransformFromCopy),
                (1,None, Indicate), (2,None, Indicate), # indicates
            ]
        ))
        self.wait()

        # distribute dots
        vdistr = MathTex(r"u_{\parallel} \cdot v_{\parallel} + u_{\perp} \cdot v_{\parallel}", font_size=65).next_to(eqr)
        AlignBaseline(vdistr, udotpvs)
        self.play(*TransformBuilder(
            udotpvs, vdistr,
            [
                ([0,0], None), ([0, 6], None), #parentheses
                ([0, slice(1,3)], [0, slice(0,2)]), #upara
                ([0,slice(7,None)],[0,slice(2, 5)], TransformFromCopy, {"path_arc":-TAU/4}), #dot vperp
                ([0, slice(3,6)], [0, slice(5,8)]), #uperp
                ([0, slice(-3, None)], [0, slice(-3,None)]) #vpara
            ]
        ))
        self.wait()

        # term to zero
        zeror = MathTex("0", font_size=65).move_to(vdistr[0][8])
        AlignBaseline(zeror, vdistr)
        self.play(
            FadeOut(vdistr[0][6:], shift=DOWN),
            FadeIn(zeror, shift=DOWN)
        )
        self.wait()

        # clean up
        upve = AlignBaseline(MathTex(r"u\cdot Pv = u_{\parallel} \cdot v_{\parallel}", font_size=65).align_to(udotpv, LEFT).shift(RIGHT*1.5), udotpv)        
        self.play(
            FadeOut(zeror, vdistr[0][5]),
            TransformMatchingShapes(VGroup(udotpv, vdistr[0][:5], eqr), upve),
            puve.animate.shift(RIGHT*1.5)
        )
        self.wait()

        # triple equality
        dotse = MathTex(r"Pu \cdot v", "=",r"u\cdot Pv", font_size=65).shift(DOWN)
        self.play(
            TransformFromCopy(puve, dotse[0]),
            Write(dotse[1]),
            TransformFromCopy(upve, dotse[2])
        )
        self.wait()

        # to transpose form
        dotst = MathTex(r"(Pu)^T v", "=",r"u^T Pv", font_size=65).shift(DOWN)
        self.play(*TransformBuilder(
            dotse, dotst,
            [
                (None, [0,0]), # open parenthesis
                ([0,slice(0,2)], [0, slice(1,3)]), # Pu
                (None, [0, 3]), # close paren
                ([0,2], [0,4]), # dot to T
                ([0,3], [0,5]), # v
                (1,1), # eq
                ([2,0], [2,0]), # u
                ([2,1], [2,1]), # dot to T
                ([2, slice(2, None)], [2, slice(2, None)]) #pv
            ]
        ))
        self.wait()

        # distribute transpose
        dotstd = AlignBaseline(MathTex(r"u^T P^T v", "=",r"u^T Pv", font_size=65).shift(DOWN), dotst)
        self.play(*TransformBuilder(
            dotst, dotstd,
            [
                ([0,0], None), ([0,3], None), #parentheses
                ([0,1], [0,2]), #p
                ([0,2], [0,0]), # u
                ([0,4], [0,1], TransformFromCopy), ([0,4], [0,3]), # T
                ([0,5], [0,4]), #v
                (1,1), (2,2), #rest
            ]
        ))
        self.wait()

        # indicate pt and p
        self.play(
            Indicate(dotstd[0][2:4]),
            Indicate(dotstd[2][2])
        )
        self.wait()

        # down to pt = p
        ptp = AlignBaseline(MathTex(r"P^T", "=",r"P", font_size=65).shift(DOWN), dotst)
        self.play(*TransformBuilder(
            dotstd, ptp,
            [
                ([0, slice(0,2)], None), ([0,4],None), #utv in first
                ([2, slice(0,2)], None), ([2,3],None), #utv in second
                (1,1), # =
                ([0,slice(2,4)], [0, slice(0,2)]), #pt
                ([2,2], [2,0]), #p
            ]
        ))
        self.wait()


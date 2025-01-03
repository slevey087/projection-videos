from manim import *
from scipy import stats
import numpy as np

from utils import *
# from manim import Arrow3D, Cone # over-writing the over-writing from utils :/

w=1


# new colors
COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN

"""
# old colors
XCOLOR = XKCD.LIGHTVIOLET
YCOLOR = XKCD.LIGHTYELLOW
VCOLOR = XKCD.LIGHTAQUA
PCOLOR = XKCD.LIGHTAQUA
RCOLOR = XKCD.LIGHTORANGE

UCOLOR = XKCD.LIGHTBLUE
# VCOLOR = XKCD.LIGHTPURPLE
PUCOLOR = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
PVCOLOR = XKCD.BLUEGREEN

"""

def color_tex_standard(equation):
    if isinstance(equation,Matrix):
        for entry in equation.get_entries(): color_tex_standard(entry)
        return equation
    return color_tex(equation,(r"\mathbf{v}", VCOLOR), (r"\mathbf{x}",XCOLOR),(r"\mathbf{y}",YCOLOR), (r"\mathbf{p}",PCOLOR),("p_x",PCOLOR),("p_y",PCOLOR), (r"\mathbf{u}", UCOLOR),(r"\hat{\mathbf{u}}", PUCOLOR),(r"\hat{\mathbf{v}}", PVCOLOR))




class ProjectionIntuition2d(MovingCameraScene):
    def construct(self):  
        camera = self.camera.frame
        camera.save_state()
        
        # first projection example     
        cone = Arc(fill_color=YELLOW)
        cone.points = np.array([
            [0,0,0],[0,0,0],[1,0,0],[1,0,0],
            *cone.points,
            [0,1,0],[0,1,0],[0,0,0],[0,0,0]
        ])
        cone.scale(2).set_opacity(1).set_stroke(0).rotate(-3*PI/4).shift(UP+RIGHT*1).set_z_index(-1).set_opacity(0.3)
        light = SVGMobject("flashlight-svgrepo-com.svg")
        light.scale(0.8).flip().rotate(-PI/4).shift(UP*3+RIGHT*1.5)
        axes = Axes(
            x_range = [-6,6,1],
            x_length = 10,
            y_range = [-3.6,3.6,1],
            y_length = 6,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False})
        v1 = Vector(axes.c2p(2,2), color=COLOR_V1)
        vp = Vector(axes.c2p(2,0), color=COLOR_V1P)#.set_opacity(0.8)

        self.play(FadeIn(axes, shift=UR), run_time=1.3)
        self.play(camera.animate.move_to(v1).scale(0.4),run_time=2)
        self.play(
            GrowArrow(v1),            
            run_time=1.5
        )
        self.wait()
        self.play(camera.animate.move_to(Group(vp,light).get_center()).scale(1.5),run_time=1.5)
        self.play(FadeIn(light))
        self.play(cone.animate(reverse_rate_function=True).scale(0,about_point=cone.get_top()),run_time=1.75) # adds cone by scaling from top down        
        self.play(TransformFromCopy(v1,vp),run_time=2)
        self.wait()

        # rotate whole thing
        line = DoubleArrow(start=axes.c2p(3,3), end=axes.c2p(-3,-3)).set_opacity(0.6).set_z_index(-1)
        line0 = DoubleArrow(start=axes.c2p(3,0), end=axes.c2p(-3,0)).set_opacity(0).set_z_index(-1)
        self.play(
            VGroup(v1, vp, light, cone).animate.rotate(PI/4, about_point=ORIGIN),            
            ReplacementTransform(line0, line),
            camera.animate.move_to(Group(light,vp).copy().rotate(PI/4,about_point=ORIGIN))
        , run_time=2)
        self.wait()

        # restore camera
        self.play(Restore(camera), run_time=2)
        self.wait()


# config.renderer="opengl"
class ProjectionIntution3d(ThreeDScene):
    def construct(self):
        vcoords = [1,2.5,2]
        grid_blocks = 4

        self.set_camera_orientation(theta=(90+17)*DEGREES, phi=75*DEGREES)
        axes = ThreeDAxes(
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False},
            z_axis_config={"include_ticks":False},
        )
        plane = OpenGLSurface(lambda u,v:[u,v,0], u_range=[-grid_blocks*axes.get_x_unit_size(),grid_blocks*axes.get_x_unit_size()], v_range=[-grid_blocks*axes.get_y_unit_size(),grid_blocks*axes.get_x_unit_size()], color=GREY).set_opacity(0.6)
        
        v1 = Arrow3D(start = ORIGIN, end=axes.c2p(*vcoords), color=COLOR_V1)
        vp = Vector(axes.c2p(*project(vcoords, [vcoords[0],vcoords[1],0])), color=COLOR_V1P)        
        cone = Cone(color=YELLOW).shift(OUT*3.25 + UP*2.75 + RIGHT*2.73).set_opacity(0)                
        self.add(cone)
        light = SVGMobject("flashlight-svgrepo-com.svg").set_opacity(0)        
        light.rotate(TAU/8).rotate(TAU/4,axis=[0,1,0]).rotate(TAU/4, axis=[1,0,0]).rotate(TAU/12, axis=[0,0,1]).scale(0.6).move_to(axes.c2p(4.75,2.75,4.25))
        light.data["stroke_width"]=np.array([0]) # necessary for a bug - spent a while on this one!
        
        self.play(FadeIn(axes,shift=OUT+UL), run_time=1.5)
        self.play(GrowFromPoint(v1,axes.get_origin()))
        self.play(FadeIn(plane), run_time=1.5)
        self.play(light.animate.set_opacity(1),run_time=1.5)
        self.play(cone.set_opacity(0.4).animate(reverse_rate_function=True).scale(0,about_point=cone.get_zenith()),run_time=1.5)
        self.play(self.camera.animate.move_to(v1).scale(0.4),run_time=2)
        self.play(ReplacementTransform(Vector(axes.c2p(*vcoords), color=COLOR_V1).set_opacity(0),vp), run_time=1.5)
        self.wait()                       



class OrthogonalProjection(MovingCameraScene):
    def construct(self):
        axes = Axes(
            x_range = [-6,6,1],
            x_length = 10,
            y_range = [-3.6,3.6,1],
            y_length = 6,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False})
        v = [ValueTracker(1), ValueTracker(4)]
        v1 = always_redraw(lambda: Vector(axes.c2p(v[0].get_value(), v[1].get_value()), color=COLOR_V1))
        vp = always_redraw(lambda: Vector(axes.c2p((v[0].get_value()*4/5+v[1].get_value()*2/5),(v[0].get_value()*2/5+v[1].get_value()*1/5)), color=COLOR_V1P))
        dash = always_redraw(lambda: DashedLine(start=v1.get_end(), end=vp.get_end(), dash_length=0.2))
        ra = always_redraw(lambda: RightAngle(vp, dash, quadrant=(-1,-1), length=0.4))        
        line = DoubleArrow(start=axes.c2p(*np.array([2,1])*2), end=axes.c2p(*np.array([2,1])*-2)).set_opacity(0.3).set_z_index(-1)
        self.play(FadeIn(axes))
        self.play(GrowArrow(v1))
        self.play(FadeIn(line,shift=DR))
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(Group(v1,vp)).scale(0.5),run_time=1.75)
        self.play(
            Write(dash),
            TransformFromCopy(v1,vp)
        ,run_time=1.5)
        self.play(self.camera.frame.animate.move_to(ra).scale(0.4),run_time=1.5)
        self.play(Write(ra))   
        self.wait()     
        self.play(
            v[0].animate.set_value(-4),
            v[1].animate.set_value(1.5),
            Restore(self.camera.frame)
        , run_time=2)
        self.play(
            v[0].animate.set_value(1),
            v[1].animate.set_value(4)
        , run_time=2)
        self.wait()

        # clear out to axes
        self.play(FadeOut(line, v1, vp, dash, ra))


class DotProduct(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,1])
        vcoords = np.array([0.5,2])
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=COLOR_V2)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=COLOR_V1)
        vhat = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=COLOR_V1P)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=COLOR_V2).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=COLOR_V1).next_to(v.get_tip(), UP)
        vhatl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=COLOR_V1P).next_to(vhat.get_tip(), DOWN,buff=0.1)
        r = DashedLine(axes.c2p(*vcoords),axes.c2p(*pcoords), dash_length=0.12).set_opacity(0.5)
        ra = RightAngle(vhat,r,length=0.25,quadrant=(-1,-1)).set_stroke(opacity=0.5)        
        diagram = VGroup(axes,x,v,xl,vl,r,vhat,vhatl,ra).shift(-VGroup(v,x).get_center()).shift(LEFT*2.25)
        diagram.remove(axes)        

        frame.scale(0.5).move_to(VGroup(x,v))

        # draw x and v
        self.play(GrowArrow(x))        
        self.play(GrowArrow(v))
        self.play(Write(xl))
        self.play(Write(vl))        
        self.wait(w)

        # move figure left, write dot product
        self.play(frame.animate.shift(RIGHT*1.5))
        dot = MathTex(r"\mathbf{x} \cdot \mathbf{v}",font_size=60).next_to(diagram).shift(UP)
        color_tex(dot,(r"\mathbf{x}",COLOR_V2),(r"\mathbf{v}",COLOR_V1),(r"\hat{\mathbf{v}}",COLOR_V1P))
        self.play(
            TransformFromCopy(xl[0][0],dot[0][0]), # x            
            TransformFromCopy(vl[0][0],dot[0][2]) # v
        ,run_time=1.5)
        self.play(FadeIn(dot[0][1],shift=UP)) # dot
        self.wait()

        # drop projection
        self.play(
            TransformFromCopy(v,vhat),
            Write(r)
        ,run_time=2)
        self.play(Write(ra))
        self.play(Write(vhatl))        
        self.wait(w)

        # dot product equation
        dot2 = AlignBaseline(MathTex(r"\mathbf{x} \cdot \mathbf{v}","=",r"|\mathbf{x}|\times |\hat{\mathbf{v}} |",font_size=60).move_to(dot),dot)
        color_tex(dot2,(r"\mathbf{x}",COLOR_V2),(r"\mathbf{v}",COLOR_V1),(r"\hat{\mathbf{v}}",COLOR_V1P))
        self.play(
            ReplacementTransform(dot[0],dot2[0],path_arc=45*DEGREES),
            Write(dot2[1]), # =
            TransformFromCopy(xl[0],dot2[2][1]), # x
            TransformFromCopy(vhatl[0],dot2[2][5:7]), # v
            Write(VGroup(*[dot2[2][i] for i in [0,2,3,4,7]]))
        ,run_time=2)
        self.wait()

        # clear out
        self.play(FadeOut(*self.mobjects,shift=DOWN))

        

class PMatrix(Scene):
    def construct(self):
        # pick up from previous
        axes = Axes(
            x_range = [-6,6,1],
            x_length = 10,
            y_range = [-3.6,3.6,1],
            y_length = 6,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False}).shift(LEFT*2.5).scale(0.75)
        

        # add vector, arrow and text
        pm = Matrix([
            [1,0],
            [0,0]
        ]).move_to([2,2,0])
        vm = Matrix([[2],[2]]).set_color(COLOR_V1).next_to(pm,RIGHT)
        va = Arrow(axes.c2p(0,0), axes.c2p(2,2), color=COLOR_V1, buff=0)
        self.play(FadeIn(axes,va,shift=DOWN))
        self.play(Write(vm))
        self.wait()

        # add matrix        
        self.play(Write(pm))
        self.wait()

        # project
        vp = Arrow(axes.c2p(0,0), axes.c2p(2,0), color=COLOR_V1P, buff=0)
        self.play(TransformFromCopy(va, vp))
        eq = MathTex("=", font_size=55).next_to(vm)
        vpm = Matrix([[2],[0]]).set_color(COLOR_V1P).next_to(eq)
        self.play(Write(eq), Write(vpm))
        self.wait()

        # write formula
        xl = MathTex("x",font_size=60,color=COLOR_V1).next_to(va.get_end(),UP)
        self.play(FadeIn(xl,shift=DOWN))
        xe = MathTex("x",font_size=65,color=COLOR_V1).next_to(vm,DOWN,buff=0.3)
        self.play(TransformFromCopy(xl,xe),run_time=1.5)
        pe = AlignBaseline(MathTex(r"\mathbf{P}",font_size=60).next_to(pm,DOWN),xe)
        self.play(TransformFromCopy(pm,pe),run_time=1.5)
        xpl = MathTex(r"\hat{x}",font_size=60,color=COLOR_V1P).next_to(vp.get_end(),DOWN)
        self.play(FadeIn(xpl,shift=UP))
        xpe = AlignBaseline(MathTex(r"\hat{x}",font_size=65,color=COLOR_V1P).next_to(vpm,DOWN),xe)
        eq2 = AlignBaseline(MathTex("=", font_size=65).next_to(eq,DOWN),xpe)        
        self.play(TransformFromCopy(xpl,xpe), Write(eq2),run_time=1.5)
        self.wait()

        # clear equation, vectors and move matrix over
        self.play(
            FadeOut(vm, eq, vpm,eq2,xe,xpe, shift=RIGHT),
            FadeOut(va, vp,xl,xpl),
            VGroup(pm,pe).animate.shift(RIGHT)
        ,run_time=1.5)
        self.wait()

        # add vectors in space
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1)
        for j in range(-3,4)] for i in range(-6,6)]
        import itertools
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)]
        )
        self.wait()

        # project them down
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,0), buff=0, color=COLOR_V1P)
        for j in range(-3,4)] for i in range(-6,6)]        
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*vectors), itertools.chain(*projected_vectors))]
        )
        self.wait()
        


class Idempotent(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.play(Write(title))
        self.play(Write(ul))
        self.wait()

        # first property
        text1 = Tex(r"Applying the projection twice is\\the same as applying it once", font_size=55)
        self.play(Write(text1))
        self.wait()
        self.play(text1.animate.next_to(ul, DOWN))
        self.wait()

        # x -> Px
        formula1 = MathTex("x",r"\longrightarrow",r"\mathbf{P}x", font_size=75).shift(DOWN)
        self.play(Write(formula1[0]))
        self.wait()
        self.play(FadeIn(formula1[1], shift=UP))
        self.play(
            Write(formula1[2][0]),
            TransformFromCopy(formula1[0][0], formula1[2][1], path_arc=-120*DEGREES)
        ,run_time=1.5)
        self.wait()

        # Px -> P(Px)
        formula2 = MathTex(r"\mathbf{P}x",r"\longrightarrow",r"\mathbf{P}(\mathbf{P}x)", font_size=75)
        formula2.shift(formula1[1].get_center()-formula2[1].get_center())
        self.play(*TransformBuilder(
            formula1, formula2,
            [
                (1,1),  # arrow
                (0, None, FadeOut, {"shift":DOWN}),  # x
                (2, 0, ReplacementTransform, {"path_arc":TAU/4}),  # Px                
            ]
        ),run_time=1.5)
        self.wait()
        self.play(
            Write(formula2[2][0]), # P
            FadeIn(formula2[2][1], formula2[2][4]),
            TransformFromCopy(formula2[0][:],formula2[2][2:4], path_arc=-120*DEGREES, run_time=1.5)
        )
        self.wait()

        # P(Px) to P^2 x
        formula3 = MathTex(r"\mathbf{P}x",r"\longrightarrow",r"\mathbf{P}^2 x", font_size=75)
        formula3.shift(formula2[1].get_center()-formula3[1].get_center())
        self.play(*TransformBuilder(
            formula2, formula3,
            [
                (slice(0,2), slice(0,2)),  # Px ->
                ([2, 0], [2,1]),  # P becomes 2
                ([2,1], None), ([2,4], None), #parentheses
                ([2,2], [2,0]),  # P
                ([2, 3], [2,2])   # x
            ]
        ),run_time=1.25)
        self.wait()

        # replace arrow with =
        formula4 = MathTex(r"\mathbf{P}x",r"=",r"\mathbf{P}^2 x", font_size=75).move_to(formula3)
        self.play(TransformMatchingTex(formula3, formula4, transform_mismatches=True),run_time=1.5)
        self.wait()

        # swap sides
        formula45 = MathTex(r"\mathbf{P}^2 x",r"=",r"\mathbf{P}x", font_size=75).move_to(formula4)
        self.play(*TransformBuilder(
            formula4, formula45,
            [
                (0,2,None,{"path_arc":120*DEGREES}), # Px
                (1,1), # = 
                (2,0,None,{"path_arc":120*DEGREES}) # p2x
            ]
        ),run_time=1.5)
        self.wait()

        # P = P^2
        formula5 = MathTex(r"\mathbf{P}^2",r"=",r"\mathbf{P}", font_size=75).move_to(formula45)
        self.play(*TransformBuilder(
            formula45, formula5,
            [
                ([2,0],[2,0]), # P
                (1,1), # =
                ([0,[0,1]],[0,[0,1]]), # P2
                ([0,2],None,FadeOut,{"shift":DOWN}), # x
                ([2,1],None,FadeOut,{"shift":DOWN}), # x
            ]            
            ), run_time=1.25
        )
        self.wait()

        # idempotent
        idem = Tex("Idempotent", font_size=55).next_to(formula5, DOWN)
        self.play(FadeIn(idem, shift=DOWN))
        self.wait()

        # increment the exponent
        forms = [formula5,*[AlignBaseline(MathTex(r"\mathbf{P}^"+k,r"=",r"\mathbf{P}", font_size=75).move_to(formula5),formula5) for k in ["3","4","5","n"]]]
        for k in range(1,len(forms)): 
            self.play(
                *[ReplacementTransform(forms[k-1][i],forms[k][i]) for i in [1,2]], # =P
                ReplacementTransform(forms[k-1][0][0], forms[k][0][0]),
                FadeOut(forms[k-1][0][-1],shift=DOWN),
                FadeIn(forms[k][0][-1],shift=DOWN)
            )
        self.wait()

        # clear out but title
        self.play(FadeOut(forms[-1], idem, text1))



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

        # p=pt, symmetric
        text2 = MathTex(r"\mathbf{P}^T=\mathbf{P}", font_size=75)
        self.play(Write(text2))
        sym = Tex("Symmetric", font_size=55).next_to(text2, DOWN)
        self.play(FadeIn(sym, shift=DOWN))
        self.wait()

        # clear out
        self.play(FadeOut(text1, text2,sym))
        self.wait()





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
        u = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=COLOR_V1)        
        v = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*vcoords), buff=0, color=COLOR_V2)                
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=COLOR_V1P)        
        pv = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pvcoords), buff=0, color=COLOR_V2P)        
        plane = OpenGLSurface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5]).set_opacity(0.4)
        # plane = Surface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5])
        diagram = Group(plane,u,v,pu,pv)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)
        diagram.shift(LEFT*2.25)
        ul = MathTex(r"\mathbf{u}", color=COLOR_V1, font_size=50).next_to(u.get_end(),buff=0.15)
        vl = MathTex(r"\mathbf{v}", color=COLOR_V2, font_size=50).next_to(v.get_end(),buff=0.15)        
        pul = MathTex(r"\hat{\mathbf{u}}", color=COLOR_V1P, font_size=50).next_to(pu.get_end(),DR,buff=0.05)
        pvl = MathTex(r"\hat{\mathbf{v}}", color=COLOR_V2P, font_size=50).next_to(pv.get_end(),DR,buff=0.05)        
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
        color_tex(dot1,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
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
        dot2 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}",r"\cdot",r"\mathbf{v}", font_size=45).move_to(dot1,aligned_edge=LEFT)
        color_tex(dot2,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
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
        dot3 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}",r"\cdot",r"\mathbf{v}","=",r"\mathbf{u}",r"\cdot",r"\hat{\mathbf{v}}", font_size=45).move_to(dot2,aligned_edge=LEFT)
        color_tex(dot3,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
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
        line = Line(axes.c2p(*(-xcoords)), axes.c2p(*xcoords), buff=0, color=GREY).set_opacity(0.5)
        u = Arrow(axes.c2p(0,0), axes.c2p(*ucoords), buff=0, color=COLOR_V1)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=COLOR_V2)
        pu = Arrow(axes.c2p(0,0), axes.c2p(*pucoords), buff=0, color=COLOR_V1P)
        pv = Arrow(axes.c2p(0,0), axes.c2p(*pvcoords), buff=0, color=COLOR_V2P)
        ul = MathTex(r"\mathbf{u}", font_size=60, color=COLOR_V1).next_to(u.get_tip(), UP)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=COLOR_V2).next_to(v.get_tip(), UP)        
        pul = MathTex(r"\hat{\mathbf{u}}", font_size=60, color=COLOR_V1P).next_to(pu.get_tip(), UL,buff=0).shift(UP*0.05+LEFT*0.1)
        pvl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=COLOR_V2P).next_to(pv.get_tip(), UR,buff=0.08)        
        diagram = VGroup(axes,line,u,v,pu,pv,ul,vl,pul,pvl).shift(-VGroup(u,v,line).get_center()).shift(UP)
        diagram.remove(axes)
        frame.scale(0.5).move_to(VGroup(line,u,v))
        
        # fade in diagram
        self.play(GrowFromCenter(line))
        for vector, label in zip([v,u],[vl,ul]):
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
        color_tex(dot1,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
        self.play(
            FadeOut(dash_u,dash_v),
            VGroup(v,vl).animate.set_opacity(0.2),            
            VGroup(u,ul).animate.set_opacity(0.2)
        ,run_time=1.5)
        self.play(
            frame.animate.scale(1.2).move_to(VGroup(diagram,dot1)),            
        )
        self.play(TransformFromCopy(pul[0],dot1[0]),run_time=1.25)
        self.play(FadeIn(dot1[1],shift=DOWN),run_time=1.25)
        self.play(TransformFromCopy(pvl[0],dot1[2]),run_time=1.5)
        self.wait(w)

        # dim vbar and add next dot product           
        dot2 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}",r"\cdot",r"\mathbf{v}", font_size=65).move_to(dot1)
        color_tex(dot2,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
        AlignBaseline(dot2,dot1)
        self.play(ReplacementTransform(dot1[:3],dot2[:3])) # first dot product            
        self.play(Write(dot2[3])) # =
        self.play(
            VGroup(pv,pvl).animate.set_opacity(0.2),       
            VGroup(u,ul).animate.set_opacity(0.2)               
        )
        self.play(VGroup(v,vl).animate.set_opacity(1),run_time=1)
        self.play(TransformFromCopy(pul[0],dot2[4]),run_time=1.5) # uhat
        self.play(Write(dot2[5])) # dot
        self.play(TransformFromCopy(vl[0],dot2[6]),run_time=1.75)
        self.wait(w)

        # to other pair                
        dot3 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}",r"\cdot",r"\mathbf{v}","=",r"\mathbf{u}",r"\cdot",r"\hat{\mathbf{v}}", font_size=65).move_to(dot2)
        color_tex(dot3,(r"\mathbf{u}",COLOR_V1),(r"\mathbf{v}",COLOR_V2),(r"\hat{\mathbf{u}}",COLOR_V1P),(r"\hat{\mathbf{v}}",COLOR_V2P))
        AlignBaseline(dot3,dot2)
        self.play(ReplacementTransform(dot2[:7],dot3[:7])) # first two dot products            
        self.play(Write(dot3[7])) # =
        self.play(
            VGroup(v,vl).animate.set_opacity(0.2),            
            VGroup(pu,pul).animate.set_opacity(0.2) 
        )
        self.play(
            VGroup(pv,pvl).animate.set_opacity(1),            
            VGroup(u,ul).animate.set_opacity(1)                                   
        , run_time=1)
        self.play(TransformFromCopy(ul[0],dot3[8]),run_time=1.75) # u
        self.play(Write(dot3[9])) # dot
        self.play(TransformFromCopy(pvl[0],dot3[10]),run_time=1.25)
        self.wait(w)



class SymmetricAlgebra(Scene):
    def construct(self):
        # u,v 
        u = MathTex("u", font_size=65).move_to(UP*1.5+LEFT*6)
        v = MathTex("v", font_size=65).move_to(DOWN*1.5+LEFT*6)
        self.play(Write(u))
        self.play(Write(v))
        self.wait()

        # extend
        ue = MathTex(r"u=\mathbf{P}u+u_{\perp}", font_size=65).move_to(u).align_to(u,LEFT)
        ve = MathTex(r"v=\mathbf{P}v+v_{\perp}", font_size=65).move_to(v).align_to(v,LEFT)
        self.play(
            ReplacementTransform(u, ue),
            ReplacementTransform(v, ve)
        ,run_time=1.5)
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

        # Pu dot v
        pudotv = AlignBaseline(MathTex(r"\mathbf{P}u \cdot v", font_size=65).next_to(ue, RIGHT*5),ue)        
        self.play(Write(pudotv))
        self.wait()
        

        # substitute into pu dot v
        eql = AlignBaseline(MathTex("=", font_size=55).next_to(pudotv, RIGHT), pudotv)
        self.play(Write(eql))
        pudotvs = MathTex(r"\mathbf{P}u \cdot \left(\mathbf{P}v+v_{\perp} \right)", font_size=65).next_to(eql, RIGHT)
        AlignBaseline(pudotvs, pudotv)        
        anims = TransformBuilder(
            [pudotv, ve, ue], pudotvs,
            [
                ([0,0,slice(0,3)], [0, slice(0,3)], TransformFromCopy,{"path_arc":-120*DEGREES}), #Pu
                (None, [0, 3]), (None, [0,-1]), # parentheses
                ([1, 0, slice(2, None)], [0, slice(-6,-1)], TransformFromCopy), # v components
                ([1,0,slice(2,None)], None, Indicate)   # indicate 
            ]
        )
        self.play(anims[0],run_time=1.25)
        self.play(anims[1:3])
        self.play(*anims[3:],run_time=1.5)
        self.wait()
        
        # distribute dots
        udistr = MathTex(r"\mathbf{P}u \cdot \mathbf{P}v + \mathbf{P}u \cdot v_{\perp}", font_size=65).next_to(eql)
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
        ),run_time=1.75)
        self.wait()
        
        # terms to zero
        zerol = MathTex("0", font_size=65).move_to(udistr[0][8])
        AlignBaseline(zerol, udistr)        
        self.play(Indicate(udistr[0][6:]))
        self.play(
            FadeOut(udistr[0][6:], shift=DOWN),
            FadeIn(zerol, shift=DOWN)
        ,run_time=1.25)
        self.wait()

        # clean up
        puve = AlignBaseline(MathTex(r"\mathbf{P}u\cdot v = \mathbf{P}u \cdot \mathbf{P}v", font_size=65).align_to(pudotv, LEFT), pudotv)        
        self.play(
            FadeOut(zerol, udistr[0][5]),
            *TransformBuilder(
                [pudotv, udistr[0],eql], puve,
                [
                    ([0,0,slice(None,None)],[0,slice(0,4)]), # pudot v
                    ([2,0,0],[0,4]), # =
                    ([1,slice(None,5)],[0,slice(5,None)]), # rest
                ]
            ),            
            run_time=1.5
        )
        self.wait()
        

        # u dot pv
        udotpv = MathTex(r"u \cdot \mathbf{P}v", font_size=65).next_to(ve, RIGHT*5)
        self.play(*TransformBuilder(
            pudotv[0],udotpv[0],
            [
                (0,2), # P
                (1,0), # u
                (2,1), # dot
                (3,3), # v
            ],TransformFromCopy
        ),run_time=1.75)
        self.wait()

        # substitute into u dot pv
        eqr = AlignBaseline(MathTex("=", font_size=55).next_to(udotpv), udotpv)
        self.play(Write(eqr),run_time=1.5)
        udotpvs = MathTex(r"\left( \mathbf{P}u + u_{\perp} \right) \cdot \mathbf{P}v", font_size=65).next_to(eqr)
        AlignBaseline(udotpvs, udotpv)
        anims = TransformBuilder(
            [udotpv, ue, ve], udotpvs,
            [
                ([0,0,slice(-3,None)], [0, slice(-3,None)], TransformFromCopy,{"path_arc":-90*DEGREES}), #dot Pv
                (None, [0, 0]), (None, [0,6]), # parentheses
                ([1, 0, slice(2, None)], [0, slice(1,6)], TransformFromCopy), # u components
                ([1,0,slice(2,None)], None, Indicate)   # indicate 
            ]
        )
        self.play(anims[0],run_time=1.75)
        self.play(anims[1:3])
        self.play(*anims[3:],run_time=1.5)
        self.wait()

        # distribute dots
        vdistr = MathTex(r"\mathbf{P}u \cdot \mathbf{P}v + u_{\perp} \cdot \mathbf{P}v", font_size=65).next_to(eqr)
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
        ),run_time=1.5)
        self.wait()

        # term to zero
        zeror = MathTex("0", font_size=65).move_to(vdistr[0][8])
        AlignBaseline(zeror, vdistr)
        self.play(Indicate(vdistr[0][6:]))
        self.play(
            FadeOut(vdistr[0][6:], shift=DOWN),
            FadeIn(zeror, shift=DOWN)
        )
        self.wait()

        # clean up
        upve = AlignBaseline(MathTex(r"u\cdot \mathbf{P}v = \mathbf{P}u \cdot \mathbf{P}v", font_size=65).align_to(udotpv, LEFT).shift(RIGHT*1.5), udotpv)        
        self.play(
            FadeOut(zeror, vdistr[0][5]),
            *TransformBuilder(
                [udotpv, vdistr[0], eqr],upve,
                [
                    ([0,0,slice(None,None)],[0,slice(0,4)]), # u dot pv
                    ([2,0,0],[0,4]), # =
                    ([1,slice(None,5)],[0,slice(5,None)]), # rest
                ]
            ),
            # TransformMatchingShapes(VGroup(udotpv, vdistr[0][:5], eqr), upve),
            puve.animate.shift(RIGHT*1.5)
        ,run_time=1.25)
        self.wait()

        # triple equality
        dotse = MathTex(r"\mathbf{P}u \cdot v", "=",r"\mathbf{P}u \cdot \mathbf{P}v","=",r"u\cdot \mathbf{P}v", font_size=85)
        self.play(
            ReplacementTransform(puve[0][:4], dotse[0][:]), # pu.v
            ReplacementTransform(puve[0][4], dotse[1][0]), # =
            Merge([puve[0][5:], upve[0][5:]],dotse[2][:]), # pu.pv
            ReplacementTransform(upve[0][:4], dotse[4][:], path_arc=120*DEGREES), # u.pv
            ReplacementTransform(upve[0][4],dotse[3][0],path_arc=90*DEGREES), # =
            FadeOut(ue,ve)
        ,run_time=2)
        self.wait()

        # remove middle equality
        dotse1 = AlignBaseline(MathTex(r"\mathbf{P}u \cdot v","=",r"u\cdot \mathbf{P}v", font_size=85),dotse)
        self.play(*TransformBuilder(
            dotse,dotse1,
            [
                (0,0), (-1,-1), # outer terms
                ([[1,3]],1,Merge), # =
                (2,None,FadeOut,{"shift":DOWN}) # inner term
            ]
        ),run_time=1.25)
        self.wait()

        # to transpose form
        dotst = AlignBaseline(MathTex(r"(\mathbf{P}u)^T v", "=",r"u^T \mathbf{P}v", font_size=85),dotse1)
        self.play(*TransformBuilder(
            dotse1, dotst,
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
        dotstd = AlignBaseline(MathTex(r"u^T \mathbf{P}^T v", "=",r"u^T \mathbf{P}v", font_size=85), dotst)
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
        ptp = AlignBaseline(MathTex(r"\mathbf{P}^T", "=",r"\mathbf{P}", font_size=85), dotst)
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



class RankDeficient(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Projecting kills some dimensions", font_size=55)
        self.play(Write(text1))
        self.wait()        

        # edit text
        text2 = Tex(r"Projection matrices are less-than-full rank", font_size=55).move_to(text1)
        self.play(FadeTransform(text1, text2[0]))
        self.wait()
        


class NSDemo(Scene):
    def construct(self):
        axes = Axes(
            x_range = [-6,6,1],
            x_length = 10,
            y_range = [-3.6,3.6,1],
            y_length = 6)        
        line = DoubleArrow(start=axes.c2p(3,3), end=axes.c2p(-3,-3)).set_opacity(0.5)
        self.add(axes, line)
        self.wait()


        # add vectors in space
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1).set_z_index(-1)
        for j in range(-3,4)] for i in range(-3,4)]
        import itertools
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)]
        )
        self.wait()

        # project them down
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=COLOR_V1P)
        for j in range(-3,4)] for i in range(-3,4)]        
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*vectors), itertools.chain(*projected_vectors))]
        )
        self.wait()
        


        # null space line
        nsline = DoubleArrow(start=axes.c2p(-3,3), end=axes.c2p(3,-3), color=RED)
        import itertools
        self.play(
            Write(nsline),
            *[FadeOut(vector) for vector in itertools.chain(*vectors)]
        )
        self.wait()

        # vectors in nullspace
        vectors = [
            Vector(axes.c2p(i,-i), color=RED_A)
        for i in range(-3,4)]
        self.play(*[GrowArrow(vector) for vector in vectors])
        self.wait()

        # do transform
        zerov = Dot(color=COLOR_V1P)
        self.play(*[
            Transform(vector, zerov)
        for vector in vectors])
        self.wait()

        # nullspace caption
        ns = Tex("Null Space",font_size=60,color=RED).next_to(nsline.get_end(),RIGHT)
        self.play(Write(ns))
        self.wait()

        # rank one caption
        r1 = Tex(r"Rank 1\\Projection", font_size=60).next_to(line.get_start())
        self.play(Write(r1))
        self.wait(w)

        # cut here, to Eigs01
        # then cut back

        # remove prior captions
        self.play(FadeOut(ns,r1))

        # eigen equation
        ee = MathTex(r"\mathbf{P}x =", r"\lambda x",font_size=75).next_to(nsline.get_start(),LEFT).shift(LEFT*0.75)
        self.play(Write(ee))
        self.wait()
        ee1 = AlignBaseline(MathTex(r"\mathbf{P}x =", "0 x",font_size=75).move_to(ee),ee)
        self.play(*TransformBuilder(
            ee, ee1,
            [
                (0,0), # px=
                ([1,0],None,FadeOut,{"shift":DOWN}), # lambda
                (None, [1,0],FadeIn,{"shift":DOWN}), # 0
                ([1,1],[1,1]), # x
            ]
        ))
        eigs0 = Tex("Eigenvalue of 0",font_size=55).next_to(ee1,DOWN,aligned_edge=LEFT)
        self.play(Write(eigs0))
        self.wait()
        ee2 = AlignBaseline(MathTex(r"\mathbf{P}x =", "0",font_size=75).move_to(ee1).align_to(ee1,LEFT),ee1)
        self.play(*TransformBuilder(
            ee1,ee2,
            [
                (0,0), # px=
                ([1,0],[1,0]), # 0
                ([1,1],None) # x
            ]
        ))
        ee = ee2

        """
        # rank caption
        rank = Tex(r"Rank of", r"\\a matrix").shift(DOWN*2).add_background_rectangle()
        eq = Tex("=").next_to(rank,RIGHT).add_background_rectangle()
        dim = Tex("How many dimensions",r"\\ its output can span").next_to(eq).align_to(rank,UP).add_background_rectangle()
        #self.play(FadeOut(nsline, zerov))
        self.play(Write(rank))
        self.play(Write(eq))
        self.play(Write(dim))
        self.wait()

        # fade out stuff
        self.play(FadeOut(rank, eq, dim, ee))"""
        
        # repeat ns demo
        # vectors in nullspace
        vectors = [
            Vector(axes.c2p(i,-i), color=COLOR_V1)
        for i in range(-3,4)]        
        self.wait()
        self.play(
            *[GrowArrow(vector) for vector in vectors],             
        )
        self.wait()

        # do transform
        zerov = Dot(color=COLOR_V1P)
        self.play(*[
            Transform(vector, zerov)
        for vector in vectors])        
        self.wait()

        # equation stuff for eig 1
        ee1 = MathTex(r"\mathbf{P}x =", r"\lambda x",font_size=75).next_to(line.get_start()).shift(RIGHT*0.75)
        self.play(Write(ee1))
        self.wait()
        ee2 = AlignBaseline(MathTex(r"\mathbf{P}x =", "1 x",font_size=75).move_to(ee1),ee1)
        self.play(*TransformBuilder(
            ee1, ee2,
            [
                (0,0), # px=
                ([1,0],None,FadeOut,{"shift":DOWN}), # lambda
                (None, [1,0],FadeIn,{"shift":DOWN}), # 1
                ([1,1],[1,1]), # x
            ]
        ))
        eigs1 = Tex("Eigenvalue of 1",font_size=55).next_to(ee2,DOWN,aligned_edge=RIGHT)
        self.play(Write(eigs1))
        self.wait()
        ee3 = AlignBaseline(MathTex(r"\mathbf{P}x =", "x",font_size=75).move_to(ee2).align_to(ee2,RIGHT),ee2)
        self.play(*TransformBuilder(
            ee2,ee3,
            [
                (0,0), # px=
                ([1,0],None), # 1
                ([1,1],[1,0]), # x
            ]
        ))
        ee1 = ee3
        

        # vectors on projected
        vectors = [
            Vector(axes.c2p(i,i), color=COLOR_V1)
        for i in range(-3,4)]        
        self.play(
            *[GrowArrow(vector) for vector in vectors],            
        )
        self.wait()

        # do transform        
        self.play(*[
            Indicate(vector, color=COLOR_V1P)
        for vector in vectors])
        self.wait()
        


class Eigs01(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Eigenvalues of projections are 0's and 1's", font_size=55)
        self.play(Write(text1))
        self.wait()                



class MultiplyProperty(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Orthogonal or overlapping projections commute", font_size=55)
        self.play(Write(text1))
        self.wait()  



# config.renderer="opengl"
class Commute(Scene):
    def construct(self):
        h = 0.6
        ucoords = np.array([0.75,0.5,h])
        pucoords = ucoords - np.array([0,0,h])
        pxucoords = np.array([ucoords[0],0,0])
        pyucoords = np.array([0,ucoords[1],0])
        vcoords = np.array([0.25,0.7,h])
        pvcoords = vcoords - np.array([0,0,h])
        

        # set up frame
        frame = self.camera
        original_frame = frame.copy()

        # define diagram
        axes = ThreeDAxes(
            x_range=[-0.5,1],x_length=5,
            y_range=[-0.5,1],y_length=5,
            z_range=[0,1],z_length=1.9,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False},
            z_axis_config={"include_ticks":False},            
        ).set_flat_stroke(False)
        
        u = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=COLOR_V1)                
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=COLOR_V1P)     
        pxu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pxucoords), buff=0, color=COLOR_V2P)                   
        pyu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pyucoords), buff=0, color=COLOR_V2P)                   
        plane = OpenGLSurface(lambda u,v:axes.c2p(*[u,v,0]),u_range=[-0.5,1],v_range=[-0.5,1]).set_opacity(0.4)
        grid = NumberPlane(
            x_range=[-0.5,1,0.25],x_length=5,
            y_range=[-0.5,1,0.25],y_length=5
        ).set_color(GRAY).set_flat_stroke(False).set_opacity(0.1)

        diagram = Group(axes, plane, grid)
        diagram.add(u,pu,pxu,pyu)       

        # dashed lines and right angles
        dxy = DashedLine(u.get_end(),pu.get_end(),dash_length=0.1).set_flat_stroke(True).set_opacity(0.5)
        rxy = RightAngleIn3D(pu,Line(pu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dx = DashedLine(u.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rx = RightAngleIn3D(pxu,Line(pxu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dxp = DashedLine(pu.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rxp = RightAngleIn3D(pxu,Line(pxu.get_end(),pu.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)        
        # for dash in [dxy,dx,dxp]: dash.reverse_points()
        diagram.add(dxy,rxy,dx,rx,dxp,rxp) 
        
        # rotate/shift the diagram instead of rotating the camera, since fixed-in-frame mobjects don't work so great for text
        diagram.rotate(-45*DEGREES).rotate(-80*DEGREES,RIGHT)
        diagram.shift(ORIGIN-pu.get_center()).shift(DOWN*0.5+LEFT*0.25)             

        # vector labels
        ul = MathTex(r"u", color=COLOR_V1, font_size=45).next_to(u.get_end(),buff=0.15)        
        pul = MathTex(r"\mathbf{P_{xy}}","u", font_size=45).next_to(pu.get_end(),RIGHT,buff=0.05)
        pul[0].set_color(COLOR_V1P), pul[1].set_color(COLOR_V1)
        pxul = MathTex(r"\mathbf{P_{x}}",r"\mathbf{P_{xy}}","u", font_size=40).next_to(pxu.get_end(),DL,buff=0.05)        
        pxul[0].set_color(COLOR_V2P), pxul[1].set_color(COLOR_V1P), pxul[2].set_color(COLOR_V1)
        diagram.add(ul,pul,pxul)   
        
        # scale the camera                
        frame.save_state()
        frame.scale(0.4)

        # add axes etc., draw vector and label it
        self.play(FadeIn(axes,plane,grid,shift=UR))
        self.play(ReplacementTransform(u.copy().scale(0.05,u.points[0]),u))        
        self.play(Write(ul))
        self.wait()

        # project to x-y plane
        self.play(
            TransformFromCopy(u,pu),
            Create(dxy),
            run_time=1.25,
        )
        self.play(Write(rxy))                
        
        # project to x axis
        self.play(
            TransformFromCopy(pu,pxu),
            Create(dxp),
            run_time=1.25,
        )
        self.play(Write(rxp)) 

        # add labels
        self.play(Write(pul),run_time=1.25)
        self.play(Write(pxul),run_time=1.25)
        self.wait()

        # zoom out
        self.play(frame.animate.scale(1.25).shift(UP*0.5),run_time=1.5)
        commute = MathTex(r"\mathbf{P_{x}}",r"\mathbf{P_{xy}}","u",r"\stackrel{?}{=}",r"\mathbf{P_{xy}}",r"\mathbf{P_{x}}","u", font_size=45).next_to(diagram,UP).shift(DOWN*0.25+RIGHT*1.5)
        commute[0].set_color(COLOR_V2P), commute[1].set_color(COLOR_V1P), commute[2].set_color(COLOR_V1), commute[4].set_color(COLOR_V1P), commute[5].set_color(COLOR_V2P), commute[6].set_color(COLOR_V1)
        self.play(TransformFromCopy(pxul[:],commute[:3]),run_time=1.5)
        self.play(Write(commute[3]))
        self.play(
            TransformFromCopy(commute[0],commute[5], path_arc=90*DEGREES), # px
            TransformFromCopy(commute[1],commute[4], path_arc=-90*DEGREES), # pxy
            TransformFromCopy(commute[2],commute[6], path_arc=90*DEGREES), # u
            run_time=2 
        )
        self.wait()

        # project straight to x axis
        pxu2 = pxu.copy()
        self.play(
            FadeOut(pxu),
            TransformFromCopy(u,pxu2),
            Create(dx),
            run_time=1.75,
        )
        self.play(Write(rx)) 

        # they do commute        
        commute1 = AlignBaseline(MathTex(r"\mathbf{P_{x}}",r"\mathbf{P_{xy}}","u",r"=",r"\mathbf{P_{xy}}",r"\mathbf{P_{x}}","u", font_size=45).next_to(diagram,UP).shift(DOWN*0.25+RIGHT*1.25).move_to(commute),commute)
        commute1[0].set_color(COLOR_V2P),commute1[1].set_color(COLOR_V1P), commute1[2].set_color(COLOR_V1), commute1[5].set_color(COLOR_V2P),commute1[4].set_color(COLOR_V1P), commute1[6].set_color(COLOR_V1)
        check = Tex(r'\checkmark', color=XKCD.LIGHTGREEN).next_to(commute1)
        self.play(ReplacementTransform(commute,commute1))
        self.play(Write(check))
        self.wait()

        # clear the canvas to just u
        self.play(FadeOut(commute1, pu,pul,pxu2,pxul,dx,rx,dxy,rxy,dxp,rxp,check),run_time=1.5)
        self.wait()

        # expression with px and py
        ncommute = MathTex(r"\mathbf{P_{y}}",r"\mathbf{P_{x}}","u","=","0","=",r"\mathbf{P_{x}}",r"\mathbf{P_{y}}","u", font_size=45).move_to(commute)
        ncommute[0].set_color(COLOR_V2P), ncommute[1].set_color(COLOR_V2P), ncommute[2].set_color(COLOR_V1), ncommute[6].set_color(COLOR_V2P), ncommute[7].set_color(COLOR_V2P), ncommute[8].set_color(COLOR_V1), 
        self.play(TransformFromCopy(ul[0],ncommute[2]),run_time=1.25)
        self.play(Write(ncommute[1]),run_time=1.25)
        self.play(Write(ncommute[0]),run_time=1.25)
        self.wait()

        # equals question
        qm = Tex(r"?").next_to(ncommute[3])
        self.play(Write(ncommute[3]))
        self.play(Write(qm))

        # project to x axis, then to zero
        self.play(
            TransformFromCopy(u,pxu2),
            Create(dx),
            Write(rx),
            run_time=1.75
        )
        self.wait()        
        self.play(
            pxu2.animate.scale(0.01,pxu2.points[0]),
            Transform(VGroup(dx,rx),Dot(axes.get_origin(),color=GREY,fill_opacity=0)),
            run_time=1.75
        )
        self.remove(pxu2)

        # replace question mark with zero        
        self.play(
            FadeOut(qm,shift=DOWN),
            FadeIn(ncommute[4],shift=DOWN)
        )
        self.wait()

        # rotate diagram, project to y axis then zero
        pyu.set_opacity(0)
        self.play(
            Group(axes,plane,grid,u,pyu).animate.rotate(-130*DEGREES,axes.z_axis.get_unit_vector()),
            ul.animate.next_to(Group(axes,plane,grid,u,pyu).copy().rotate(-130*DEGREES,axes.z_axis.get_unit_vector())[3].points[-1],LEFT,buff=0.25),
            run_time=2
        )
        self.remove(pyu)
        pyu.set_opacity(1)
        self.play(TransformFromCopy(u,pyu),run_time=1.75)
        self.wait()        
        self.play(pyu.animate.scale(0.01,pyu.points[0]),run_time=1.75)
        self.remove(pyu)

        # add second formula
        self.play(Write(ncommute[5]))
        self.play(
            TransformFromCopy(ncommute[0],ncommute[7], path_arc=90*DEGREES), # py
            TransformFromCopy(ncommute[1],ncommute[6], path_arc=-90*DEGREES), # px
            TransformFromCopy(ncommute[2],ncommute[8], path_arc=90*DEGREES), # u
            run_time=2 
        )
        self.wait()

        self.interactive_embed()


class NotCommute(MovingCameraScene):
    def construct(self):
        l1coords = np.array([2,0.5])*2        
        l2coords = np.array([0.5,2])*2                
        ucoords = np.array([1.5,1.5])*2
        p1coords = l1coords * np.dot(l1coords,ucoords) / np.dot(l1coords,l1coords)
        p12coords = l2coords * np.dot(l2coords,p1coords) / np.dot(l2coords,l2coords)
        p2coords = l2coords * np.dot(l2coords,ucoords) / np.dot(l2coords,l2coords)
        p21coords = l1coords * np.dot(l1coords,p2coords) / np.dot(l1coords,l1coords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)#.rotate(-15*DEGREES)
        line1 = Line(axes.c2p(*(-l1coords)), axes.c2p(*l1coords), buff=0, color=GREY).set_opacity(0.4)
        line2 = Line(axes.c2p(*(-l2coords)), axes.c2p(*l2coords), buff=0, color=GREY).set_opacity(0.4)
        u = Arrow(axes.c2p(0,0), axes.c2p(*ucoords), buff=0, color=COLOR_V1)
        diagram = VGroup(axes,line1,line2,u)

        # projections
        p1 = Arrow(axes.c2p(0,0), axes.c2p(*p1coords), buff=0, color=COLOR_V1P)
        p21 = Arrow(axes.c2p(0,0), axes.c2p(*p21coords), buff=0, color=COLOR_V1P)
        p2 = Arrow(axes.c2p(0,0), axes.c2p(*p2coords), buff=0, color=COLOR_V2P)
        p12 = Arrow(axes.c2p(0,0), axes.c2p(*p12coords), buff=0, color=COLOR_V2P)
        diagram.add(p1,p12,p21,p2)

        # labels
        ul = MathTex(r"u", font_size=60, color=COLOR_V1).next_to(u.get_tip(), UP)
        p1l = MathTex(r"\mathbf{P_1}","u", font_size=60, color=COLOR_V1P).next_to(p1.get_tip(), DOWN)        
        p21l = MathTex(r"\mathbf{P_1}",r"\mathbf{P_2}", "u", font_size=60, color=COLOR_V1P).next_to(p21.get_tip(), DOWN)        
        p2l = MathTex(r"\mathbf{P_2}","u", font_size=60, color=COLOR_V2P).next_to(p2.get_tip(), LEFT)        
        p12l = MathTex(r"\mathbf{P_2}",r"\mathbf{P_1}", "u", font_size=60, color=COLOR_V2P).next_to(p12.get_tip(), LEFT)        
        for label in [ul,p1l,p21l,p2l,p12l]:
            color_tex(label,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        diagram.add(ul,p1l,p12l,p21l,p2l)

        # dashes and right angles
        d1 = DashedLine(u.get_end(),p1.get_end(),dash_length=0.2).set_opacity(0.4)
        r1 = RightAngle(d1,p1,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)
        d2 = DashedLine(u.get_end(),p2.get_end(),dash_length=0.2).set_opacity(0.4)
        r2 = RightAngle(d2,p2,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)
        d12 = DashedLine(p1.get_end(),p12.get_end(),dash_length=0.2).set_opacity(0.4)
        r12 = RightAngle(d12,p12,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)
        d21 = DashedLine(p2.get_end(),p21.get_end(),dash_length=0.2).set_opacity(0.4)
        r21 = RightAngle(d21,p21,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)
        diagram.add(d1,r1,d2,r2,d12,r12,d21,r21)

        diagram.shift(-u.get_center())
        frame.scale(0.65)

        # draw lines and vector
        self.play(Create(line1), Create(line2),run_time=1.25)
        self.play(GrowArrow(u))
        self.play(Write(ul))
        self.wait()

        # project 1...
        self.play(
            TransformFromCopy(u,p1),
            Write(d1),
            TransformFromCopy(ul[0],p1l[1],path_arc=60*DEGREES),
            run_time=1.25
        )
        self.play(            
            Write(p1l[0]),
            Write(r1),
            run_time=1.25
        )
        
        # ...then 2
        self.play(
            TransformFromCopy(p1,p12),
            Write(d12),
            TransformFromCopy(p1l[:],p12l[1:],path_arc=120*DEGREES),
            run_time=1.5
        )
        self.play(
            Write(p12l[0]),
            Write(r12),
            run_time=1.25
        )
        self.wait()

        # project 2...
        self.play(
            TransformFromCopy(u,p2),
            Write(d2),
            TransformFromCopy(ul[0],p2l[1],path_arc=-60*DEGREES),
            run_time=1.25
        )
        self.play(            
            Write(p2l[0]),
            Write(r2),
            run_time=1.25
        )

        # ...then 1
        self.play(
            TransformFromCopy(p2,p21),
            Write(d21),
            TransformFromCopy(p2l[:],p21l[1:],path_arc=120*DEGREES),
            run_time=1.5
        )
        self.play(
            Write(p21l[0]),
            Write(r21),
            run_time=1.25
        )
        self.wait()

        # zoom out
        self.play(frame.animate.scale(1.2).shift(DOWN*0.5+RIGHT), run_time=1.25)        

        # write equation
        ncommute = MathTex(r"\mathbf{P_1}",r"\mathbf{P_2}",r"\neq",r"\mathbf{P_2}",r"\mathbf{P_1}", font_size=70).next_to(p21l,DR,buff=0.4)
        color_tex(ncommute,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        self.play(TransformFromCopy(p21l[:2],ncommute[:2]),run_time=1.25)
        self.play(Write(ncommute[2]))
        self.play(TransformFromCopy(p12l[:2],ncommute[3:],path_arc=120*DEGREES),run_time=1.75)        
        self.wait()

        # fade to black
        self.play(FadeOut(*self.mobjects))



class AddingMatrices(Scene):
    def construct(self):
        # p1 + p2
        f1 = MathTex(r"\mathbf{P_1}+\mathbf{P_2}",font_size=75)
        color_tex(f1,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        self.play(Write(f1[0][0:2]))
        self.play(Write(f1[0][2]))
        self.play(Write(f1[0][3:]))
        self.wait()

        # add parentheses and u
        f2 = AlignBaseline(MathTex(r"\left(",r"\mathbf{P_1}+\mathbf{P_2}",r"\right)","u",font_size=75).move_to(f1),f1)
        color_tex(f2,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        self.play(*TransformBuilder(
            f1,f2,
            [
                (None,0), # (
                (0,1), # p1+p2
                (None, 2), # )
                (None,3), # u
            ]
        ),run_time=1.25)
        self.wait()

        # = and distribute
        f3 = AlignBaseline(MathTex(r"\left(",r"\mathbf{P_1}+\mathbf{P_2}",r"\right)","u","=",r"\mathbf{P_1} u+\mathbf{P_2} u",font_size=75).move_to(f2),f2)
        color_tex(f3,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        self.play(ReplacementTransform(f2[0:4],f3[0:4]))
        self.play(Write(f3[4]))
        self.play(*TransformBuilder(
            f3,f3,
            [
                ([1,[0,1]],[5,[0,1]],None,{"path_arc":120*DEGREES}), # p1
                ([1,2],[5,3],None,{"path_arc":120*DEGREES}), # +
                ([1,[3,4]],[5,[4,5]],None,{"path_arc":120*DEGREES}), # p2
                ([3,0],[5,2],None,{"path_arc":-120*DEGREES}), # u
                ([3,0],[5,6],None,{"path_arc":-120*DEGREES}), # u
            ],TransformFromCopy
        ),run_time=1.5)
        self.wait()

        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.play(Write(title), FadeOut(f3))
        self.play(Write(ul))
        self.wait()

        # text property 1
        text1 = Tex(r"Orthogonal and non-overlapping projections \\ add to another projection", font_size=55)
        self.play(Write(text1))
        self.wait()  

        # text property 2
        text2 = Tex(r"A complete set of such projections \\ adds to the Identity Matrix", font_size=55).shift(DOWN*1.25)
        self.play(text1.animate.shift(UP*1.25))
        self.play(Write(text2))
        self.wait()  


# config.renderer="opengl"
class AddingExamples(Scene):
    def construct(self):
        h = 0.6
        ucoords = np.array([0.75,0.5,h])
        pucoords = ucoords - np.array([0,0,h])
        pxucoords = np.array([ucoords[0],0,0])
        pyucoords = np.array([0,ucoords[1],0])
        pzucoords = np.array([0,0,ucoords[2]])
        vcoords = np.array([0.25,0.7,h])
        pvcoords = vcoords - np.array([0,0,h])
        

        # set up frame
        frame = self.camera
        original_frame = frame.copy()

        # define diagram
        axes = ThreeDAxes(
            x_range=[-0.5,1],x_length=5,
            y_range=[-0.5,1],y_length=5,
            z_range=[0,1],z_length=1.9,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False},
            z_axis_config={"include_ticks":False},            
        ).set_flat_stroke(False)
        
        u = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=COLOR_V1)                
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=COLOR_V1P)     
        pxu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pxucoords), buff=0, color=COLOR_V2P)                   
        pyu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pyucoords), buff=0, color=COLOR_V2P)                   
        pzu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pzucoords), buff=0, color=COLOR_V2P)                   
        yforadd = Arrow3D(axes.c2p(*pxucoords), axes.c2p(*pucoords), buff=0, color=COLOR_V2P)                   
        plane = OpenGLSurface(lambda u,v:axes.c2p(*[u,v,0]),u_range=[-0.5,1],v_range=[-0.5,1]).set_opacity(0.4)
        grid = NumberPlane(
            x_range=[-0.5,1,0.25],x_length=5,
            y_range=[-0.5,1,0.25],y_length=5
        ).set_color(GRAY).set_flat_stroke(False).set_opacity(0.1)

        diagram = Group(axes, plane, grid)
        diagram.add(u,pu,pxu,pyu,yforadd,pzu)       

        # dashed lines and right angles
        dxy = DashedLine(u.get_end(),pu.get_end(),dash_length=0.1).set_flat_stroke(True).set_opacity(0.5)
        rxy = RightAngleIn3D(pu,Line(pu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dx = DashedLine(u.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rx = RightAngleIn3D(pxu,Line(pxu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dy = DashedLine(u.get_end(),pyu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.7)
        ry = RightAngleIn3D(pyu,Line(pyu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dz = DashedLine(u.get_end(),pzu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rz = RightAngleIn3D(pzu,Line(pzu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dxp = DashedLine(pu.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rxp = RightAngleIn3D(pxu,Line(pxu.get_end(),pu.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)        
        # for dash in [dxy,dx,dxp]: dash.reverse_points()
        diagram.add(dxy,rxy,dx,rx,dy,ry,dz,rz) 
        
        # rotate/shift the diagram instead of rotating the camera, since fixed-in-frame mobjects don't work so great for text
        diagram.rotate(-145*DEGREES).rotate(-78*DEGREES,RIGHT)
        diagram.shift(ORIGIN-pu.get_center()).shift(DOWN*0.5+LEFT*0.25)             

        # vector labels
        ul = MathTex(r"u", color=COLOR_V1, font_size=45).next_to(u.get_end(),UP,buff=0.15)        
        pul = MathTex(r"\mathbf{P_{xy}}","u", font_size=45).next_to(pu.get_end(),UR,buff=0.05).shift(RIGHT*0.5)
        pul[0].set_color(COLOR_V1P),pul[1].set_color(COLOR_V1)
        pxul = MathTex(r"\mathbf{P_{x}}","u", font_size=40).next_to(pxu.get_end(),UP,buff=0.05)        
        pyul = MathTex(r"\mathbf{P_{y}}","u", font_size=40).next_to(pyu.get_end(),RIGHT,buff=0.05)        
        pzul = MathTex(r"\mathbf{P_{z}}","u", font_size=40).next_to(pzu.get_end(),RIGHT,buff=0.05)        
        for label in [pxul,pyul,pzul]:
            label[0].set_color(COLOR_V2P), label[1].set_color(COLOR_V1)
        diagram.add(ul,pul,pxul,pyul,pzul)   

        
        # scale the camera                
        frame.save_state()
        frame.scale(0.4)
    
        # add frame, draw u
        self.play(FadeIn(axes,plane,grid,shift=UR))
        self.play(ReplacementTransform(u.copy().scale(0.05,u.points[0]),u))        
        self.play(Write(ul))
        self.wait()

        # project to x          
        self.play(
            TransformFromCopy(u,pxu),
            Create(dx),
            run_time=1.5,
        )
        self.play(
            Write(rx),
            Write(pxul),
            run_time=1.5
        ) 

        # project to y
        self.play(
            TransformFromCopy(u,pyu),
            Create(dy),
            run_time=1.5,
        )
        self.play(
            Write(ry),
            Write(pyul),
            run_time=1.5
        ) 

        # add x and y
        self.play(TransformFromCopy(pyu,yforadd))
        self.play(ReplacementTransform(pu.copy().scale(0.05,pu.points[0]),pu))        
        self.play(FadeOut(yforadd))
        self.wait()

        # project to xy
        self.play(FadeOut(pu))
        self.play(
            TransformFromCopy(u,pu),
            Create(dxy),
            run_time=1.5,
        )
        self.play(
            Write(rxy),
            Write(pul),
            run_time=1.5
        ) 
        self.wait()

        # zoom out, show formula
        self.play(frame.animate.scale(1.15).move_to([0,.702,0]),run_time=1.5)
        f1 = MathTex(r"\mathbf{P_x}","u","+",r"\mathbf{P_y}","u","=",r"\mathbf{P_{xy}}","u").next_to(diagram,UP).shift(DOWN*0.1)        
        f1[0].set_color(COLOR_V2P), f1[1].set_color(COLOR_V1), f1[3].set_color(COLOR_V2P), f1[4].set_color(COLOR_V1), f1[6].set_color(COLOR_V1P), f1[7].set_color(COLOR_V1)
        self.play(
            TransformFromCopy(pxul[:],f1[:2]), # pxu
            Write(f1[2]), #+
            TransformFromCopy(pyul[:],f1[3:5]), # pyu
            run_time=1.5
        )
        self.play(Write(f1[5])), # =
        self.play(TransformFromCopy(pul[:],f1[6:]))
        self.wait()

        # drop u from formula
        f2 = AlignBaseline(MathTex(r"\mathbf{P_x}","+",r"\mathbf{P_y}","=",r"\mathbf{P_{xy}}").move_to(f1),f1)
        f2[0].set_color(COLOR_V2P), f2[2].set_color(COLOR_V2P),f2[4].set_color(COLOR_V1P),
        self.play(*TransformBuilder(
            f1,f2,
            [
                (0,0), # px
                (1,None,FadeOut,{"shift":DOWN}), # u
                (2,1), # +
                (3,2), #py
                (4,None,FadeOut,{"shift":DOWN}), # u
                (5,3), # =
                (6,4), # pxy
                (7,None,FadeOut,{"shift":DOWN}), # u
            ]
        ))
        self.wait()

        # project onto z
        self.play(
            TransformFromCopy(u,pzu),
            Create(dz),
            run_time=1.5,
        )
        self.play(
            Write(rz),
            Write(pzul),
            run_time=1.5
        ) 

        # add z projection to formula, to I
        f3 = AlignBaseline(MathTex(r"\mathbf{P_x}","+",r"\mathbf{P_y}","+",r"\mathbf{P_z}","=",r"I").move_to(f2),f2)
        f3[0].set_color(COLOR_V2P), f3[2].set_color(COLOR_V2P), f3[4].set_color(COLOR_V2P)
        self.play(*TransformBuilder(
            f2,f3,
            [
                (0,0), # px
                (1,1), #+
                (2,2), #py
                (None,3), # +
                # pz below
                (3, 5),#=
                (4,None,FadeOut,{"shift":DOWN}), # pxy
                (None,6,FadeIn,{"shift":DOWN}), # I
            ]
            ),
            TransformFromCopy(pzul[0],f3[4]),
            run_time=1.5
        )
        self.wait()



class NotAdd(MovingCameraScene):
    def construct(self):
        l1coords = np.array([2,0.5])*2        
        l2coords = np.array([0.5,2])*2                
        ucoords = np.array([0.5,1])*2
        p1coords = l1coords * np.dot(l1coords,ucoords) / np.dot(l1coords,l1coords)        
        p2coords = l2coords * np.dot(l2coords,ucoords) / np.dot(l2coords,l2coords)
        sumcoords = p1coords+p2coords        
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # draw vectors and labels
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)#.rotate(-15*DEGREES)
        line1 = Line(axes.c2p(*(-l1coords)), axes.c2p(*l1coords), buff=0, color=GREY).set_opacity(0.4)
        line2 = Line(axes.c2p(*(-l2coords)), axes.c2p(*l2coords), buff=0, color=GREY).set_opacity(0.4)
        u = Arrow(axes.c2p(0,0), axes.c2p(*ucoords), buff=0, color=COLOR_V1)
        diagram = VGroup(axes,line1,line2,u)

        # projections
        p1 = Arrow(axes.c2p(0,0), axes.c2p(*p1coords), buff=0, color=COLOR_V1P)        
        p2 = Arrow(axes.c2p(0,0), axes.c2p(*p2coords), buff=0, color=COLOR_V2P)
        padd = Arrow(axes.c2p(0,0), axes.c2p(*sumcoords), buff=0, color=COLOR_V2)
        diagram.add(p1,p2,padd)

        # labels
        ul = MathTex(r"u", font_size=55).next_to(u.get_tip(), UP)
        p1l = MathTex(r"\mathbf{P_1}","u", font_size=55).next_to(p1.get_tip(), DOWN)                
        p2l = MathTex(r"\mathbf{P_2}","u", font_size=55).next_to(p2.get_tip(), LEFT)        
        paddl = MathTex(r"\left(",r"\mathbf{P_1}","+",r"\mathbf{P_2}",r"\right)", "u", font_size=60).next_to(padd.get_tip())        
        for label in [ul,p1l,p2l,paddl]:
            color_tex(label,("u",COLOR_V1),(r"\mathbf{P_1}",COLOR_V1P),(r"\mathbf{P_2}",COLOR_V2P))
        diagram.add(ul,p1l,p2l,paddl)

        # dashes and right angles
        d1 = DashedLine(u.get_end(),p1.get_end(),dash_length=0.2).set_opacity(0.4)
        r1 = RightAngle(d1,p1,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)
        d2 = DashedLine(u.get_end(),p2.get_end(),dash_length=0.2).set_opacity(0.4)
        r2 = RightAngle(d2,p2,length=0.2,quadrant=(-1,-1)).set_stroke(opacity=0.4)        
        diagram.add(d1,r1,d2,r2)

        # starting frame
        diagram.shift(-padd.get_center())
        frame.scale(0.5)

        # draw lines and vector
        self.play(Create(line1), Create(line2),run_time=1.25)
        self.play(GrowArrow(u))
        self.play(Write(ul))
        self.wait()

        # project 1
        self.play(
            TransformFromCopy(u,p1),
            Write(d1),
            TransformFromCopy(ul[0],p1l[1],path_arc=60*DEGREES),
            run_time=1.5
        )
        self.play(            
            Write(p1l[0]),
            Write(r1),
            run_time=1.5
        )
        
        # project 2
        self.play(
            TransformFromCopy(u,p2),
            Write(d2),
            TransformFromCopy(ul[0],p2l[1],path_arc=-60*DEGREES),
            run_time=1.5
        )
        self.play(            
            Write(p2l[0]),
            Write(r2),
            run_time=1.5
        )

        # add projections
        add = p2.copy().put_start_and_end_on(p1.get_end(),padd.get_end())
        self.play(TransformFromCopy(p2,add),run_time=1.5)
        self.play(GrowArrow(padd),run_time=1.5)
        self.play(
            FadeOut(add),
            frame.animate.scale(1.2).shift(RIGHT*0.75)
        )
        self.play(
            Write(paddl[0]), # (
            TransformFromCopy(p1l[0], paddl[1],path_arc=-120*DEGREES), # p1
            Write(paddl[2]), # +
            TransformFromCopy(p2l[0],paddl[3],path_arc=120*DEGREES), # p2
            Write(paddl[4]), # )
            Merge([p1l[1].copy(),p2l[1].copy()],paddl[5],animargs={"path_arc":-120*DEGREES}), # u
            run_time=2
        )
        self.wait()

        # fade to black
        self.play(FadeOut(*self.mobjects))


class TraceIsRank(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.play(Write(title))
        self.play(Write(ul))
        self.wait()

        # text property 1
        text1 = MathTex(r"\text{trace}(\mathbf{P})","=",r"\text{rank}(\mathbf{P})", font_size=65)
        for part in text1: self.play(Write(part))
        self.wait()  

        # shift up
        self.play(text1.animate.next_to(ul,DOWN))

        # add matrix and take its trace
        m = Matrix([
            [0.5,-0.5,0],
            [-0.5,0.5,0],
            [0,0,1]
        ],element_alignment_corner=ORIGIN)
        mo = MobjectMatrix([[m]],left_bracket="(", right_bracket=")").shift(DOWN)
        trace = MathTex(r"\text{trace}",font_size=60).next_to(mo,LEFT)        
        self.play(Create(mo),run_time=1.25)                
        self.play(Write(trace))
        sum = MathTex("=","0.5","+","0.5","+","1","=","2",font_size=60).next_to(mo)
        VGroup(trace.generate_target(),mo.generate_target(),sum).arrange().center().shift(DOWN)
        self.play(MoveToTarget(trace),MoveToTarget(mo))
        self.play(Write(sum[0]))
        self.play(
            *[Indicate(m.get_entries()[i],scale_factor=1.6,color=COLOR_V1) for i in [0,4,8]],
            TransformFromCopy(m.get_entries()[0],sum[1]), # 0.5
            TransformFromCopy(m.get_entries()[4],sum[3]), # -0.5
            TransformFromCopy(m.get_entries()[8],sum[5]), # 1
            Write(sum[2]), Write(sum[4]), # +'s
            run_time=1.5
        )
        self.play(Write(sum[6])) # =
        self.play(Merge([sum[1].copy(),sum[3].copy(),sum[5].copy()],sum[7]),run_time=1.5), # 2
        self.wait()

        # clear all that out
        self.play(FadeOut(trace,mo,sum))

        # trace is sum of eigenvalues
        trse = Tex("trace"," = ",r"$\lambda_1 + \lambda_2+...+\lambda_n$",font_size=65)
        for part in trse: self.play(Write(part))
        self.wait()

        # eigs are 0 and 1
        proje = Tex(r"For projection,\\",r"$\lambda$'s"," = ","1's" ," and ","0's",font_size=65).next_to(trse,DOWN).shift(DOWN)
        color_tex(proje,("1",COLOR_V1),("0",RED))
        for part in proje: self.play(Write(part))
        self.wait()

        # add rank(p) before 1's
        proje1 = Tex(r"For projection,\\",r"$\lambda$'s"," = ",r"rank$(\mathbf{P})$ ","1's" ," and ","0's",font_size=65).move_to(proje)
        color_tex(proje1,("1",COLOR_V1),("0",RED))
        self.play(*TransformBuilder(
            proje,proje1,
            [
                (slice(0,3),slice(0,3)), # up to =
                (None,3,FadeIn,{"shift":UP}), # rank(p)
                (slice(3,None),slice(4,None)), # 1's to end
            ]
        ),run_time=1.5)
        self.wait()

        # add rank(p) before 1's
        proje2 = Tex(r"For projection,\\",r"$\lambda$'s"," = ",r"rank$(\mathbf{P})$ ","1's" ," and ",r"$\left(n-\text{rank}(\mathbf{P})\right)$ ","0's",font_size=65).move_to(proje1)
        color_tex(proje2,("1",COLOR_V1),("0",RED))
        self.play(*TransformBuilder(
            proje1,proje2,
            [
                (slice(0,6),slice(0,6)), # up to and
                (None,6,FadeIn,{"shift":UP}), # n-rank(p)
                (6,7), # 0s
            ]
        ),run_time=1.5)
        self.wait()

        # substitue in for trace, to read trace = rank(p)\cdot 1 + (n-rank(p))\cdot0 = rank(p)
        trse2 = Tex("trace"," = ",r"rank($\mathbf{P}$)",r"$\cdot$", "$1$","$+$",r"$\left(n-\text{rank}(\mathbf{P}) \right)$",r"$\cdot$","$0$",font_size=65).move_to(trse)
        color_tex(trse2,("$1$",COLOR_V1),("$0$",RED))
        self.play(*TransformBuilder(
            [trse,proje2],trse2,
            [
                ([0,slice(0,2)],slice(0,2)), # trace =
                ([0,2],5), # l1+l2..+ln to +
                ([1,slice(0,3)],None), # for projection ls = 
                ([1,5],None), # and
                ([1,3],2), # rank p
                (None,3), # cdot
                ([1,4],4), # 1s to 1
                ([1,6],6), # n-rankp
                (None,7), # cdot
                ([1,7],8), # 0s to 0
            ]
        ),run_time=2)
        self.wait()

        # down to just rank p
        trse3 = Tex("trace"," = ",r"rank($\mathbf{P}$)",font_size=65).move_to(trse2)
        self.play(
            ReplacementTransform(trse2[:2],trse3[:2]), # trace=
            Merge([trse2[2],trse2[3:]],trse3[2]) # rankp
            ,run_time=1.5)
        self.wait()

        # down to the title
        self.play(FadeOut(trse3,text1))
        self.wait()



class ProjectionPropsRecap(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=BLUE)        
        self.add(title, ul)
        self.wait()

        # properties
        props = VGroup(*[Tex(message, font_size=55).next_to(ul, DOWN) for message in [
            r"Idempotent: $\mathbf{P}^2=\mathbf{P}$",
            r"Symmetric: $\mathbf{P}^T=\mathbf{P}$",
            r"Less-than-full rank: $\text{rank}(\mathbf{P})<n$",
            r"Eigenvalues are 0's and 1's",
            r"$\mathbf{P_2}\mathbf{P_1}=\mathbf{P_1}\mathbf{P_2}$ if orthogonal or overlapping",
            r"Orthogonal \& Non-overlapping projections add"
        ]])                
        for i in range(1,len(props)): AlignBaseline(props[i].align_to(props[0], LEFT), props[0]).shift(DOWN*i)
        props.next_to(ul,DOWN)
        for prop in props:
            color_tex(prop,(r"$\mathbf{P_1}$",COLOR_V1P),(r"$\mathbf{P_2}$",COLOR_V2P),("0",RED),("1",COLOR_V1))
            self.play(Write(prop))
            self.wait()


        

"""

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
        # self.play(Write(xl))
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
"""
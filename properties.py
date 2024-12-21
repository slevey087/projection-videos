from manim import *
from scipy import stats
import numpy as np

from utils import *
# from manim import Arrow3D, Cone # over-writing the over-writing from utils :/

w=1

XCOLOR = XKCD.LIGHTVIOLET
YCOLOR = XKCD.LIGHTYELLOW
VCOLOR = XKCD.LIGHTAQUA
PCOLOR = XKCD.LIGHTAQUA
RCOLOR = XKCD.LIGHTORANGE

UCOLOR = XKCD.LIGHTBLUE
# VCOLOR = XKCD.LIGHTPURPLE
PUCOLOR = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
PVCOLOR = XKCD.BLUEGREEN


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
        v1 = Vector(axes.c2p(2,2), color=BLUE_B)
        vp = Vector(axes.c2p(2,0), color=BLUE_E)#.set_opacity(0.8)

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
        
        v1 = Arrow3D(start = ORIGIN, end=axes.c2p(*vcoords), color=BLUE_B)
        vp = Vector(axes.c2p(*project(vcoords, [vcoords[0],vcoords[1],0])), color=BLUE_E)        
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
        self.play(ReplacementTransform(Vector(axes.c2p(*vcoords), color=BLUE_B).set_opacity(0),vp), run_time=1.5)
        self.wait()                       



"""class ProjectionIntution3d(ThreeDScene):
    def construct(self):
        vcoords = [1,2.5,2]
        grid_blocks = 4

        self.set_camera_orientation(theta=0.3, phi=1.3)
        axes = ThreeDAxes(
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False},
            z_axis_config={"include_ticks":False},
        )
        plane = Surface(lambda u,v:[u,v,0], u_range=[-grid_blocks*axes.get_x_unit_size(),grid_blocks*axes.get_x_unit_size()], v_range=[-grid_blocks*axes.get_y_unit_size(),grid_blocks*axes.get_x_unit_size()], resolution=grid_blocks*2, checkerboard_colors=[RED_E, RED_D]).set_z_index(-1)
        
        v1 = Arrow3D(start = ORIGIN, end=axes.c2p(*vcoords), color=BLUE_B)
        vp = Vector(axes.c2p(*project(vcoords, [vcoords[0],vcoords[1],0])), color=BLUE_E)
        light = SVGMobject("flashlight-svgrepo-com.svg")
        light.rotate(TAU/8).rotate(TAU/4,axis=[0,1,0]).rotate(TAU/4, axis=[1,0,0]).rotate(TAU/12, axis=[0,0,1]).scale(0.6).move_to(axes.c2p(4.75,2.75,4.25))
        cone = Cone(fill_color=YELLOW).set_z_index(-1).shift(OUT*3.25, UP*2.75, RIGHT*3.25).set_opacity(0.4)
        
        self.play(Create(axes))
        self.play(Create(v1))
        self.play(FadeIn(plane))
        self.play(FadeIn(light))
        self.play(Create(cone))
        self.play(FadeIn(vp, shift=IN))
        self.wait()                       

"""


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
        v1 = always_redraw(lambda: Vector(axes.c2p(v[0].get_value(), v[1].get_value()), color=BLUE_B))
        vp = always_redraw(lambda: Vector(axes.c2p((v[0].get_value()*4/5+v[1].get_value()*2/5),(v[0].get_value()*2/5+v[1].get_value()*1/5)), color=BLUE_E))
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
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        vhat = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)
        vhatl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=PCOLOR).next_to(vhat.get_tip(), DOWN,buff=0.1)
        r = DashedLine(axes.c2p(*vcoords),axes.c2p(*pcoords), dash_length=0.12).set_opacity(0.5)
        ra = RightAngle(vhat,r,length=0.25,quadrant=(-1,-1))        
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
        color_tex_standard(dot)
        self.play(
            TransformFromCopy(xl[0][0],dot[0][0]), # x
            Write(dot[0][1]), # dot
            TransformFromCopy(vl[0][0],dot[0][2]) # v
        ,run_time=1.5)
        self.wait()

        # drop projection
        self.play(
            TransformFromCopy(v,vhat),
            Write(r)
        ,run_time=2)
        self.play(Write(vhatl))        
        self.wait(w)

        # dot product equation
        dot2 = AlignBaseline(MathTex(r"\mathbf{x} \cdot \mathbf{v}","=",r"|\mathbf{x}|\times |\hat{\mathbf{v}} |",font_size=60).move_to(dot),dot)
        color_tex_standard(dot2)
        self.play(
            ReplacementTransform(dot[0],dot2[0],path_arc=45*DEGREES),
            Write(dot2[1]), # =
            TransformFromCopy(xl[0],dot2[2][1]), # x
            TransformFromCopy(vhatl[0],dot2[2][5:7]), # v
            Write(VGroup(*[dot2[2][i] for i in [0,2,3,4,7]]))
        ,run_time=2)
        self.wait()

        

class PMatrix(Scene):
    def construct(self):
        # pick up from previous
        axes = Axes(
            x_range = [-6,6,1],
            x_length = 10,
            y_range = [-3.6,3.6,1],
            y_length = 6,
            x_axis_config={"include_ticks":False},
            y_axis_config={"include_ticks":False})
        self.add(axes)      
        self.wait()

        # shift axes over
        self.play(
            Transform(axes, axes.copy().shift(LEFT*2.5).scale(0.75))
        )
        self.wait()

        # add vector, arrow and text
        pm = Matrix([
            [1,0],
            [0,0]
        ]).move_to([2,2,0])
        vm = Matrix([[2],[2]]).set_color(BLUE_B).next_to(pm,RIGHT)
        va = Arrow(axes.c2p(0,0), axes.c2p(2,2), color=BLUE_B, buff=0)
        self.play(GrowArrow(va))
        self.play(Write(vm))
        self.wait()

        # add matrix        
        self.play(Write(pm))
        self.wait()

        # project
        vp = Arrow(axes.c2p(0,0), axes.c2p(2,0), color=BLUE_E, buff=0)
        self.play(TransformFromCopy(va, vp))
        eq = MathTex("=", font_size=55).next_to(vm)
        vpm = Matrix([[2],[0]]).set_color(BLUE_E).next_to(eq)
        self.play(Write(eq), Write(vpm))
        self.wait()

        # write formula
        xl = MathTex("x",font_size=60,color=BLUE_B).next_to(va.get_end(),UP)
        self.play(FadeIn(xl,shift=DOWN))
        xe = MathTex("x",font_size=65,color=BLUE_B).next_to(vm,DOWN,buff=0.3)
        self.play(TransformFromCopy(xl,xe),run_time=1.5)
        pe = AlignBaseline(MathTex(r"\mathbf{P}",font_size=60).next_to(pm,DOWN),xe)
        self.play(TransformFromCopy(pm,pe),run_time=1.5)
        xpl = MathTex(r"\hat{x}",font_size=60,color=BLUE_E).next_to(vp.get_end(),DOWN)
        self.play(FadeIn(xpl,shift=UP))
        xpe = AlignBaseline(MathTex(r"\hat{x}",font_size=65,color=BLUE_E).next_to(vpm,DOWN),xe)
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
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=BLUE_B)
        for j in range(-3,4)] for i in range(-6,6)]
        import itertools
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)]
        )
        self.wait()

        # project them down
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,0), buff=0, color=BLUE_E)
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
        formula1 = MathTex("x",r"\longrightarrow","Px", font_size=75).shift(DOWN)
        self.play(Write(formula1[0]))
        self.wait()
        self.play(FadeIn(formula1[1], shift=UP))
        self.play(
            Write(formula1[2][0]),
            TransformFromCopy(formula1[0][0], formula1[2][1], path_arc=-120*DEGREES)
        ,run_time=1.5)
        self.wait()

        # Px -> P(Px)
        formula2 = MathTex("Px",r"\longrightarrow","P(Px)", font_size=75)
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
        formula3 = MathTex("Px",r"\longrightarrow","P^2x", font_size=75)
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
        formula4 = MathTex("Px",r"=","P^2x", font_size=75).move_to(formula3)
        self.play(TransformMatchingTex(formula3, formula4, transform_mismatches=True),run_time=1.5)
        self.wait()

        # swap sides
        formula45 = MathTex("P^2x",r"=","Px", font_size=75).move_to(formula4)
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
        formula5 = MathTex("P^2",r"=","P", font_size=75).move_to(formula45)
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
        forms = [formula5,*[AlignBaseline(MathTex("P^"+k,r"=","P", font_size=75).move_to(formula5),formula5) for k in ["3","4","5","n"]]]
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
        text2 = MathTex("P^T=P", font_size=75)
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
        u = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=UCOLOR)        
        v = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*vcoords), buff=0, color=VCOLOR)                
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=PUCOLOR)        
        pv = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pvcoords), buff=0, color=PVCOLOR)        
        plane = OpenGLSurface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5]).set_opacity(0.4)
        # plane = Surface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5])
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
        line = Line(axes.c2p(*(-xcoords)), axes.c2p(*xcoords), buff=0, color=XCOLOR).set_opacity(0.4)
        u = Arrow(axes.c2p(0,0), axes.c2p(*ucoords), buff=0, color=UCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        pu = Arrow(axes.c2p(0,0), axes.c2p(*pucoords), buff=0, color=PUCOLOR)
        pv = Arrow(axes.c2p(0,0), axes.c2p(*pvcoords), buff=0, color=PVCOLOR)
        ul = MathTex(r"\mathbf{u}", font_size=60, color=UCOLOR).next_to(u.get_tip(), UP)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        pul = MathTex(r"\hat{\mathbf{u}}", font_size=60, color=PUCOLOR).next_to(pu.get_tip(), UL,buff=0.05)
        pvl = MathTex(r"\hat{\mathbf{v}}", font_size=60, color=PVCOLOR).next_to(pv.get_tip(), DOWN,buff=0.15)        
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
        color_tex_standard(dot1)
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
        dot2 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}", font_size=65).move_to(dot1)
        color_tex_standard(dot2)
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
        dot3 = MathTex(r"\hat{\mathbf{u}}",r"\cdot",r"\hat{\mathbf{v}}","=",r"\hat{\mathbf{u}}","\cdot",r"\mathbf{v}","=",r"\mathbf{u}","\cdot",r"\hat{\mathbf{v}}", font_size=65).move_to(dot2)
        color_tex_standard(dot3)
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
        self.play(
            FadeOut(udistr[0][6:], shift=DOWN),
            FadeIn(zerol, shift=DOWN)
        ,run_time=1.25)
        self.wait()

        # clean up
        puve = AlignBaseline(MathTex(r"\mathbf{P}u\cdot v = \mathbf{P}u \cdot \mathbf{P}v", font_size=65).align_to(pudotv, LEFT), pudotv)        
        self.play(
            FadeOut(zerol, udistr[0][5]),
            TransformMatchingShapes(VGroup(pudotv, udistr[0][:5], eql), puve)
        ,run_time=1.5)
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
        self.play(
            FadeOut(vdistr[0][6:], shift=DOWN),
            FadeIn(zeror, shift=DOWN)
        )
        self.wait()

        # clean up
        upve = AlignBaseline(MathTex(r"u\cdot \mathbf{P}v = \mathbf{P}u \cdot \mathbf{P}v", font_size=65).align_to(udotpv, LEFT).shift(RIGHT*1.5), udotpv)        
        self.play(
            FadeOut(zeror, vdistr[0][5]),
            TransformMatchingShapes(VGroup(udotpv, vdistr[0][:5], eqr), upve),
            puve.animate.shift(RIGHT*1.5)
        ,run_time=1.25)
        self.wait()

        # triple equality
        dotse = MathTex(r"\mathbf{P}u \cdot v", "=",r"\mathbf{P}u \cdot \mathbf{P}v","=",r"u\cdot \mathbf{P}v", font_size=75)
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
        dotse1 = AlignBaseline(MathTex(r"\mathbf{P}u \cdot v","=",r"u\cdot \mathbf{P}v", font_size=75),dotse)
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
        dotst = AlignBaseline(MathTex(r"(\mathbf{P}u)^T v", "=",r"u^T \mathbf{P}v", font_size=75),dotse1)
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
        dotstd = AlignBaseline(MathTex(r"u^T \mathbf{P}^T v", "=",r"u^T \mathbf{P}v", font_size=75), dotst)
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
        ptp = AlignBaseline(MathTex(r"\mathbf{P}^T", "=",r"\mathbf{P}", font_size=75), dotst)
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
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=BLUE_B).set_z_index(-1)
        for j in range(-3,4)] for i in range(-3,4)]
        import itertools
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)]
        )
        self.wait()

        # project them down
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=BLUE_E)
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
        zerov = Dot(color=BLUE_E)
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
            Vector(axes.c2p(i,-i), color=BLUE_B)
        for i in range(-3,4)]        
        self.wait()
        self.play(
            *[GrowArrow(vector) for vector in vectors],             
        )
        self.wait()

        # do transform
        zerov = Dot(color=BLUE_E)
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
            Vector(axes.c2p(i,i), color=BLUE_B)
        for i in range(-3,4)]        
        self.play(
            *[GrowArrow(vector) for vector in vectors],            
        )
        self.wait()

        # do transform        
        self.play(*[
            Indicate(vector, color=BLUE_E)
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
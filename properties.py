from manim import *
from scipy import stats
import numpy as np

from utils import *
# from manim import Arrow3D, Cone # over-writing the over-writing from utils :/

w=1

from sklearn.linear_model import LinearRegression


# new colors
COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN
COLOR_V3P = XKCD.ELECTRICPURPLE # XKCD.EASTERPURPLE and XKCD.LIGHTISHPURPLE are good too

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

class Thumbnail(MovingCameraScene):
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
        xl = MathTex(r"\mathbf{v}", font_size=60, color=COLOR_V2).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{x}", font_size=60, color=COLOR_V1).next_to(v.get_tip(), UP,buff=0.17).shift(0.05*RIGHT)
        vhatl = MathTex(r"\hat{\mathbf{x}}", font_size=60, color=COLOR_V1P).next_to(vhat.get_tip(), DOWN,buff=0.1)
        r = DashedLine(axes.c2p(*vcoords),axes.c2p(*pcoords), dash_length=0.12).set_opacity(0.5)
        ra = RightAngle(vhat,r,length=0.25,quadrant=(-1,-1)).set_stroke(opacity=0.5)        
        diagram = VGroup(axes,x,v,xl,vl,r,vhat,vhatl,ra).shift(-VGroup(v,x).get_center()).shift(LEFT*2.25)

        diagram.remove(xl)
        diagram.remove(axes)        

        frame.scale(0.5).move_to(VGroup(x,v)).shift(UP*.4+LEFT*0.3)

        self.add(diagram)

        # write title
        title = Tex("Projection Matrix Properties", font_size=45).move_to(frame).align_to(frame,UP).shift(DOWN*0.2)
        ul = Line(title.get_corner(DL)+DL*0.07, title.get_corner(DR)+DR*0.07, color=COLOR_V1)        
        self.add(title, ul)


class Thumbnail2(MovingCameraScene):
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
        xl = MathTex(r"\mathbf{v}", font_size=50, color=COLOR_V2).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{x}", font_size=50, color=COLOR_V1).next_to(v.get_tip(), UP,buff=0.17).shift(0.05*RIGHT)
        vhatl = MathTex(r"\hat{\mathbf{x}}", font_size=50, color=COLOR_V1P).next_to(vhat.get_tip(), DOWN,buff=0.1)
        r = DashedLine(axes.c2p(*vcoords),axes.c2p(*pcoords), dash_length=0.12).set_opacity(0.5)
        ra = RightAngle(vhat,r,length=0.25,quadrant=(-1,-1)).set_stroke(opacity=0.5)        
        diagram = VGroup(axes,x,v,xl,vl,r,vhat,vhatl,ra).shift(-VGroup(v,x).get_center()).shift(LEFT*2.25)

        diagram.remove(xl)
        diagram.remove(axes)        

        frame.scale(0.5).move_to(VGroup(x,v)).shift(UP*.4+RIGHT*1.3)

        self.add(diagram)    

        # write title
        title = Tex("Projection Matrix Properties", font_size=45).move_to(frame).align_to(frame,UP).shift(DOWN*0.2)
        ul = Line(title.get_corner(DL)+DL*0.07, title.get_corner(DR)+DR*0.07, color=COLOR_V1)        
        self.add(title, ul)

        # projection properties
        p1 = MathTex(r"\mathbf{P}^2 = \mathbf{P}",font_size=55)
        p2 = MathTex(r"\mathbf{P}^T = \mathbf{P}",font_size=55).next_to(p1,DOWN)
        color_tex(p1,r"\mathbf{P}",COLOR_V1P)
        color_tex(p2,r"\mathbf{P}",COLOR_V1P)
        dots = MathTex(r"\vdots").next_to(p2,DOWN)
        VGroup(p1,p2,dots).next_to(diagram).shift(RIGHT*0.6)
        self.add(p1,p2,dots)
        




class Opening(MovingCameraScene):
    def construct(self):
        l1coords = np.array([2,0.5])*2        
        l2coords = np.array([0.5,2])*1.9                
        ucoords = np.array([1.5,1.5])*2
        p1coords = l1coords * np.dot(l1coords,ucoords) / np.dot(l1coords,l1coords)
        p12coords = l2coords * np.dot(l2coords,p1coords) / np.dot(l2coords,l2coords)
        p2coords = l2coords * np.dot(l2coords,ucoords) / np.dot(l2coords,l2coords)
        p21coords = l1coords * np.dot(l1coords,p2coords) / np.dot(l1coords,l1coords)
        
        # initial frame stuff
        frame = self.camera.frame        

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
        diagram.remove(axes)
        frame.scale(0.65)
        frame.save_state()

        # draw lines and vector
        self.play(            
            frame.animate(run_time=1.5).move_to(u).scale(0.75),
            Create(line1,run_time=1.5), 
            Create(line2,run_time=1.5),
            GrowArrow(u,run_time=1.5),
            )

        # project 1...
        self.play(
            frame.animate.move_to(p1.get_end()+LEFT*0.65).scale(2/3),
            TransformFromCopy(u,p1),
            Write(d1),            
            Write(r1),
            run_time=1.75
        )
        
        
        # ...then 2
        self.play(
            frame.animate.move_to(p12.get_end()+DOWN*0.6),
            TransformFromCopy(p1,p12),
            Write(d12),
            Write(r12),
            run_time=1.7
        )        
        

        # project 2...        
        self.play(            
            frame.animate.move_to(p2.get_end()+DOWN*0.7),
            TransformFromCopy(u,p2),
            Write(d2),
            Write(r2),
            run_time=1.5
        )
        

        # ...then 1
        self.play(
            frame.animate.move_to(p21.get_end()+UL*0.5),
            TransformFromCopy(p2,p21),
            Write(d21),
            Write(r21),
            run_time=1.75
        )

        # zoom out        
        title = Tex("Projection Matrix Properties", font_size=50).next_to(VGroup(u,p1,p2),UP).shift(UP*0.3+RIGHT*0.1)
        ul = Line(title.get_corner(DL)+DL*0.1, title.get_corner(DR)+DR*0.1, color=COLOR_V1)        
        self.play(frame.animate.scale(2).move_to(title).align_to(title,UP).shift(UP*0.35),run_time=1.75)
        self.play(
            Write(title),
            Write(ul)
        )        
        self.wait()

        self.play(
            VGroup(title,ul).animate.move_to(frame),
            FadeOut(diagram),
            run_time=1.5
        )
        self.wait()
        





class OLSdemo(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[-1, 6],
            y_range=[-1, 6],
            axis_config={"include_tip": True},
            x_length=6,
            y_length=6
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        
        # Generate random data points
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(0, 5, 20).reshape(-1, 1)
        y = 1 + 0.8 * X + np.random.normal(0, 0.4, (20, 1))
        
        # Fit linear regression
        reg = LinearRegression().fit(X, y)
        
        # Create dots for data points
        dots = VGroup(*[
            Dot(axes.c2p(X[i, 0], y[i, 0]), radius=0.05)
            for i in range(len(X))
        ])
        
        # Create regression line
        x_range = np.array([0, 5]).reshape(-1, 1)
        y_pred = reg.predict(x_range)
        line = Line(
            start=axes.c2p(x_range[0, 0], y_pred[0,0]),
            end=axes.c2p(x_range[1, 0], y_pred[1,0]),
            color=COLOR_V1
        )
        
        # Animate
        self.play(Create(axes), Create(x_label), Create(y_label))
        
        self.play(AnimationGroup(
            *[GrowFromCenter(dot) for dot in dots],
            lag_ratio=0.05
        ))
        
        self.play(Create(line))
        self.wait()




class PerspectiveProjection(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.camera.frame_center = np.array([0, 0, 0])
        
        # Create a cube
        cube = Cube(side_length=2, fill_opacity=0.7, fill_color=BLUE)
        cube.move_to(np.array([0, 0, 0]))
        
        # Create camera point (represented as a small sphere)
        camera_point = Sphere(radius=0.1, fill_color=RED)
        camera_point.move_to(np.array([4, 4, 4]))
        
        # Create viewing plane (semi-transparent rectangle)
        viewing_plane = Rectangle(
            height=4, width=4,
            fill_color=GREEN,
            fill_opacity=0.2,
            stroke_width=1
        ).rotate(angle=PI/4, axis=UP)
        viewing_plane.move_to(np.array([2, 2, 2]))
        
        # Animation sequence
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Step 1: Show the cube
        self.play(Create(cube))
        self.wait(0.5)
        
        # Step 2: Add camera point
        self.play(Create(camera_point))
        self.wait(0.5)
        
        # Step 3: Add viewing plane
        self.play(Create(viewing_plane))
        self.wait(0.5)
        
        # Step 4: Add projection lines
        projection_lines = VGroup()
        vertices = [
            np.array([x, y, z]) 
            for x in [-1, 1] 
            for y in [-1, 1] 
            for z in [-1, 1]
        ]

        for vertex in vertices:
            start_point = vertex
            end_point = camera_point.get_center()
            line = Line(
                start=start_point,
                end=end_point,
                color=YELLOW                
            ).set_opacity(0.4)
            projection_lines.add(line)
                
        self.play(Create(projection_lines))
        self.wait(1)
        
        # Step 5: Rotate cube to show perspective changes
        self.play(
            Rotate(cube, angle=PI/2, axis=UP),
            run_time=2
        )
        self.wait(1)
        
        self.stop_ambient_camera_rotation()



class PCADemo(ThreeDScene):
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



class OLSdemo(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[-1, 6],
            y_range=[-1, 6],
            axis_config={"include_tip": True},
            x_length=6,
            y_length=6
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        
        # Generate random data points
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(0, 5, 20).reshape(-1, 1)
        y = 1 + 0.8 * X + np.random.normal(0, 0.4, (20, 1))
        
        # Fit linear regression
        reg = LinearRegression().fit(X, y)
        
        # Create dots for data points
        dots = VGroup(*[
            Dot(axes.c2p(X[i, 0], y[i, 0]), radius=0.05)
            for i in range(len(X))
        ])
        
        # Create regression line
        x_range = np.array([0, 5]).reshape(-1, 1)
        y_pred = reg.predict(x_range)
        line = Line(
            start=axes.c2p(x_range[0, 0], y_pred[0,0]),
            end=axes.c2p(x_range[1, 0], y_pred[1,0]),
            color=COLOR_V1
        )
        
        # Animate
        self.play(Create(axes), Create(x_label), Create(y_label))
        
        self.play(AnimationGroup(
            *[GrowFromCenter(dot) for dot in dots],
            lag_ratio=0.05
        ))
        
        self.play(Create(line))
        self.wait()




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

        # clear out
        self.play(FadeOut(*self.mobjects, shift=DL))
        


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

        # clear out
        self.play(FadeOut(*self.mobjects))


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
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
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
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Orthogonal projection matrices are symmetric", font_size=55)
        self.play(Write(text1))
        self.wait()
        self.play(text1.animate.next_to(ul, DOWN))
        

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

        # un-dim all the things
        self.play(
            VGroup(v,vl,pu,pul).animate.set_opacity(1)
        )
        self.wait()



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
        self.wait()
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
        self.wait()
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
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
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
        


class NSDemo(MovingCameraScene):
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

        # add vectors in space, but red for nullspace
        non_ns_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1).set_z_index(-1)
        for j in range(-3,4) if j!=-i] for i in range(-3,4)]
        ns_vectors = [
            Vector(axes.c2p(i,-i), color=RED_A)
        for i in range(-3,4)]        
        import itertools
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*non_ns_vectors)],
            *[GrowArrow(vector) for vector in ns_vectors]
        )

        # zoom in
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(axes @ (-2,2)).scale(0.3),run_time=1.75)

        # project them down
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=COLOR_V1P)
        for j in range(-3,4) if j!=-i] for i in range(-3,4)]        
        zerov = Dot(color=RED_A)
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*non_ns_vectors), itertools.chain(*projected_vectors))],
            *[Transform(vector,zerov) for vector in ns_vectors],
            self.camera.frame.animate.move_to(axes @ (0,0)),
            run_time=2
        )
        self.play(
            *[FadeOut(vector) for vector in itertools.chain(*non_ns_vectors)],
            *[vector.animate.set_opacity(0) for vector in ns_vectors],
            Restore(self.camera.frame),
            run_time=1.5

        )
        self.remove(*ns_vectors)
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
        self.wait()
        
        # add vectors back, with our example in front
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1).set_z_index(-1)
        for j in range(-3,4)] for i in range(-3,4)]        
        vectors[4][0].set_z_index(2)
        self.play(
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)]
        )
        
        # zoom in and color one example vector
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.move_to(vectors[4][0].get_end()*0.75).scale(0.2),
            vectors[4][0].animate.set_color(COLOR_V3P),
            run_time=2.5
        )
        self.wait()

        # project them down and zoom in
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=COLOR_V1P)
        for j in range(-3,4)] for i in range(-3,4)]        
        projected_vectors[4][0].set_color(COLOR_V3P).set_z_index(2)
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*vectors), itertools.chain(*projected_vectors))],
            self.camera.frame.animate.move_to(projected_vectors[4][0]),
            run_time=3
        )
        self.wait()

        # fade out vectors
        self.play(*[FadeOut(vector) for vector in itertools.chain(*vectors)],run_time=1.25)
        self.wait()

        # grow new vectors and move camera        
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1).set_z_index(-1)
        for j in range(-3,4)] for i in range(-3,4)]        
        vectors[5][5].set_z_index(2)
        self.play(            
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)],
            self.camera.frame.animate.move_to(vectors[5][5].get_end()*0.75),
            run_time=3
        )
        self.wait()

        # color vector purple
        self.play(            
            vectors[5][5].animate.set_color(COLOR_V3P),
            run_time=2
        )
        self.wait()

        # project vectors
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=COLOR_V1P)
        for j in range(-3,4)] for i in range(-3,4)]        
        projected_vectors[5][5].set_color(COLOR_V3P).set_z_index(2)
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*vectors), itertools.chain(*projected_vectors))],            
            run_time=2
        )
        self.wait()


        # equation stuff for eig 1
        ee1 = MathTex(r"\mathbf{P}x =", r"\lambda x",font_size=22).next_to(vectors[5][4].get_end()).shift(RIGHT*0.17+DOWN*0.22)
        self.play(Write(ee1))
        self.wait()
        ee2 = AlignBaseline(MathTex(r"\mathbf{P}x =", "1 x",font_size=22).move_to(ee1),ee1)
        self.play(*TransformBuilder(
            ee1, ee2,
            [
                (0,0), # px=
                ([1,0],None,FadeOut,{"shift":DOWN}), # lambda
                (None, [1,0],FadeIn,{"shift":DOWN}), # 1
                ([1,1],[1,1]), # x
            ]
        ))
        eigs1 = Tex("Eigenvalue of 1",font_size=22).next_to(ee2,DOWN,aligned_edge=RIGHT,buff=0.1)
        self.play(Write(eigs1))
        self.wait()
        ee3 = AlignBaseline(MathTex(r"\mathbf{P}x =", "x",font_size=22).move_to(ee2).align_to(ee2,RIGHT),ee2)
        self.play(*TransformBuilder(
            ee2,ee3,
            [
                (0,0), # px=
                ([1,0],None), # 1
                ([1,1],[1,0]), # x
            ]
        ))
        ee1 = ee3
        self.wait()
        
        # fade text and vectors and zoom out
        self.play(
            FadeOut(ee1,eigs1),
            *[FadeOut(vector) for vector in itertools.chain(*vectors)],
            Restore(self.camera.frame),
            run_time=2
        )

        # grow new vectors and zoom in again        
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=COLOR_V1).set_z_index(-1)
        for j in range(-3,4)] for i in range(-3,4)]        
        vectors[5][1].set_z_index(2)
        self.play(            
            *[GrowArrow(vector) for vector in itertools.chain(*vectors)],
            self.camera.frame.animate.move_to(vectors[5][1].get_end()*0.55).scale(0.25),
            run_time=2
        )

        # color vector purple
        self.play(            
            vectors[5][1].animate.set_color(COLOR_V3P),
            run_time=2
        )

        # project vectors
        projected_vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(*project([i,j], [1,1])), buff=0, color=COLOR_V1P)
        for j in range(-3,4)] for i in range(-3,4)]        
        projected_vectors[5][1].set_color(COLOR_V3P).set_z_index(2)
        self.play(
            *[Transform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*vectors), itertools.chain(*projected_vectors))],            
            run_time=2
        )
        self.wait()


        # eigen equation
        ee = MathTex(r"\mathbf{P}x =", r"\lambda x",font_size=25).next_to(axes @ (1,-1)).shift(RIGHT*0.3)
        self.play(Write(ee))
        self.wait()
        ee1 = AlignBaseline(MathTex(r"\mathbf{P}x =", "0 x",font_size=25).move_to(ee),ee)
        self.play(*TransformBuilder(
            ee, ee1,
            [
                (0,0), # px=
                ([1,0],None,FadeOut,{"shift":DOWN}), # lambda
                (None, [1,0],FadeIn,{"shift":DOWN}), # 0
                ([1,1],[1,1]), # x
            ]
        ))
        eigs0 = Tex("Eigenvalue of 0",font_size=25).next_to(ee1,UP,aligned_edge=RIGHT,buff=0.15)
        self.play(Write(eigs0))
        self.wait()
        ee2 = AlignBaseline(MathTex(r"\mathbf{P}x =", "0",font_size=25).move_to(ee1).align_to(ee1,LEFT),ee1)
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
        


class Eigs01(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Eigenvalues of projections are 0's and 1's", font_size=55)
        self.play(Write(text1))
        self.wait() 

        # fade out text
        self.play(FadeOut(text1,shift=DOWN))               
        self.wait()



class MultiplyProperty(Scene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
        self.add(title, ul)
        self.wait()

        # text property
        text1 = Tex(r"Orthogonal, nested, and \\  orthogonally-overlapping \\ projections commute", font_size=60)
        self.play(Write(text1))
        self.wait()  

        # fade text
        # self.play(text1.animate.next_to(ul,DOWN))        
        self.play(FadeOut(text1))

        # vector multiplication
        eq = MathTex(r"\mathbf{P_2}",r"\mathbf{P_1}","u", font_size=75)
        eq[0].set_color(COLOR_V2P), eq[1].set_color(COLOR_V1P), eq[2].set_color(COLOR_V1)
        self.play(Write(eq[2]))
        self.play(FadeIn(eq[1],shift=RIGHT))
        t1 = Tex("Do this projection first").next_to(eq[1][0],UP).shift(UP*1.25)
        a1 = Arrow(t1.get_bottom(),eq[1][0],color=COLOR_V1P)
        t2 = Tex("Then this one").next_to(eq[0][0],DOWN).shift(DOWN*1.25)
        a2 = Arrow(t2.get_top(),eq[0][0], color=COLOR_V2P)
        self.play(
            Write(t1),
            GrowArrow(a1)
        )
        self.play(FadeIn(eq[0],shift=RIGHT))
        self.play(
            Write(t2),
            GrowArrow(a2)
        )
        self.wait()

        # other order
        eq2 = MathTex(r"\mathbf{P_2}",r"\mathbf{P_1}","u","=",r"\mathbf{P_1}",r"\mathbf{P_2}","u", font_size=75)
        eq2[0].set_color(COLOR_V2P), eq2[1].set_color(COLOR_V1P), eq2[2].set_color(COLOR_V1)
        eq2[5].set_color(COLOR_V2P), eq2[4].set_color(COLOR_V1P), eq2[6].set_color(COLOR_V1)
        self.play(
            *TransformBuilder(
                eq, eq2,
                [
                    (0,0), (1,1), (2,2), # LHS
                    (None,3, FadeIn,{"shift":RIGHT}), # =
                    (0,5,TransformFromCopy,{"path_arc":150*DEGREES}), # p2
                    (1,4,TransformFromCopy,{"path_arc":-250*DEGREES}), # p1
                    (2,6,TransformFromCopy,{"path_arc":180*DEGREES}), # u
                ]
            ),
            FadeOut(a1,a2,t1,t2),
            run_time=2.25
        )

        # remove u
        eq3 = MathTex(r"\mathbf{P_2}",r"\mathbf{P_1}","=",r"\mathbf{P_1}",r"\mathbf{P_2}", font_size=75)
        eq3[0].set_color(COLOR_V2P), eq3[1].set_color(COLOR_V1P)
        eq3[4].set_color(COLOR_V2P), eq3[3].set_color(COLOR_V1P)
        self.play(*TransformBuilder(
            eq2,eq3,
            [
                (0,0),(1,1), #LHS
                (2,None,FadeOut,{"shift":DOWN}), #u left
                (3,2), # =
                (4,3), (5,4), # rhs
                (6,None,FadeOut,{"shift":DOWN}), #u right
            ]
            ),
            run_time=1.25
        )

        # commute
        com = Tex("Commute", font_size=55).next_to(eq3,DOWN)
        self.play(FadeIn(com,shift=DOWN))
        self.wait()

        # exclamation point
        point = SVGMobject("exclamation-round-svgrepo-com.svg")
        point.scale(1.5).set_color(XKCD.LIGHTRED).set_opacity(0.9)#.move_to(VGroup(eq3,com))
        self.play(ReplacementTransform(point.copy().scale(2).set_opacity(0),point))
        self.wait()

        # to projections commute
        text2 = Tex(r"Commuting projections will \\ yield another projection", font_size=60)
        self.play(
            FadeOut(point, eq3, com),
            Write(text2)
        )
        self.wait()

        # move title up
        self.wait()
        self.play(text2.animate.next_to(ul,DOWN))

        # write PQ
        eq1 = MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2",font_size=75).shift(DOWN)
        eq1[1].set_color(COLOR_V1P), eq1[2].set_color(COLOR_V2P)
        self.play(
            FadeIn(eq1[1],shift=RIGHT),
            FadeIn(eq1[2],shift=LEFT),
        )
        self.wait()

        # add squared
        self.play(
            FadeIn(eq1[0],shift=RIGHT),
            FadeIn(eq1[3],shift=LEFT),
        )
        self.wait()

        # is idempotent question mark
        eq15 = AlignBaseline(MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2",r"\stackrel{?}{=}",r"\mathbf{P}",r"\mathbf{Q}",font_size=75).move_to(eq1),eq1) 
        for i in [1,5]: eq15[i].set_color(COLOR_V1P)
        for i in [2,6]: eq15[i].set_color(COLOR_V2P)
        self.play(ReplacementTransform(eq1[:4],eq15[:4])) # (pq)2            
        self.play(
            ReplacementTransform(eq15[4].copy().scale(6).set_opacity(0), eq15[4]), # eq
            TransformFromCopy(eq1[1:3],eq15[5:7],path_arc=-120*DEGREES),  # pq
            run_time=1.25
        )
        self.wait()

        # expand squared
        eq2 = AlignBaseline(MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2","=",r"\mathbf{P}",r"\mathbf{Q}",r"\mathbf{P}",r"\mathbf{Q}",font_size=75).move_to(eq15),eq15)
        for i in [1,5,7]: eq2[i].set_color(COLOR_V1P)
        for i in [2,6,8]: eq2[i].set_color(COLOR_V2P)        
        self.play(*TransformBuilder(
            eq15,eq2,
            [
                (slice(0,4),slice(0,4)), # (PQ)2
                (4,4), # =
                (slice(5,None),None,FadeOut,{"shift":DOWN}), # pq
            ]
        ),run_time=1.25)
        self.play(*TransformBuilder(
            eq15,eq2,
            [
                (1,5,TransformFromCopy, {"path_arc":120*DEGREES}), # P
                (2,6,TransformFromCopy, {"path_arc":120*DEGREES}), # P
                (1,7,TransformFromCopy, {"path_arc":-120*DEGREES}), # P
                (2,8,TransformFromCopy, {"path_arc":-120*DEGREES}), # P
            ],            
        ),run_time=1.5)
        self.wait()

        # commute terms
        eq3 = AlignBaseline(MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2","=",r"\mathbf{P}",r"\mathbf{P}",r"\mathbf{Q}",r"\mathbf{Q}",font_size=75).move_to(eq2),eq2)
        for i in [1,5,6]: eq3[i].set_color(COLOR_V1P)
        for i in [2,7,8]: eq3[i].set_color(COLOR_V2P)
        self.play(*TransformBuilder(
            eq2,eq3,
            [
                (slice(0,6),slice(0,6)), # pq2 = p
                (6,7,None,{"path_arc":240*DEGREES}), # Q goes right
                (7,6,None,{"path_arc":240*DEGREES}), # P goes left
                (8,8) # q
            ]
        ),run_time=1.35)
        self.wait()

        # collect to squares
        eq4 = AlignBaseline(MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2","=",r"\mathbf{P}",r"^2",r"\mathbf{Q}",r"^2",font_size=75).move_to(eq3),eq3)
        for i in [1,5]: eq4[i].set_color(COLOR_V1P)
        for i in [2,7]: eq4[i].set_color(COLOR_V2P)
        self.play(*TransformBuilder(
            eq3,eq4,
            [
                (slice(0,6),slice(0,6)), # (pq)2 = p
                (6,6), # p to 2
                (7,7), # q
                (8,8), # q to 2
            ]
        ),run_time=1.25)
        self.wait()

        # merge squareds
        eq5 = AlignBaseline(MathTex("(",r"\mathbf{P}",r"\mathbf{Q}",")^2","=",r"\mathbf{P}",r"\mathbf{Q}",font_size=75).move_to(eq4),eq4)
        for i in [1,5]: eq5[i].set_color(COLOR_V1P)
        for i in [2,6]: eq5[i].set_color(COLOR_V2P)
        self.play(
            ReplacementTransform(eq4[:5], eq5[:5]), # up to =
            Merge([eq4[5],eq4[6]],eq5[5]), # p2 to p
            Merge([eq4[7],eq4[8]],eq5[6]), # q2 to q
            run_time=1.25
        )
        self.wait()

        c = MathTex(r"\checkmark",font_size=180).set_color(XKCD.MINTYGREEN).next_to(eq5)
        self.play(ReplacementTransform(c.copy().center().scale(5).set_opacity(0),c))
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
        pxzucoords = np.array([ucoords[0],0,ucoords[2]])
        

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
        pu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=COLOR_V2P)     
        pxu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pxucoords), buff=0, color=COLOR_V1P)                   
        pyu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pyucoords), buff=0, color=COLOR_V2P)                   
        pxzu = Arrow3D(axes.c2p(*ORIGIN), axes.c2p(*pxzucoords), buff=0, color=COLOR_V3P)                   
        plane = OpenGLSurface(lambda u,v:axes.c2p(*[u,v,0]),u_range=[-0.5,1],v_range=[-0.5,1]).set_opacity(0.4)
        grid = NumberPlane(
            x_range=[-0.5,1,0.25],x_length=5,
            y_range=[-0.5,1,0.25],y_length=5
        ).set_color(GRAY).set_flat_stroke(False).set_opacity(0.1)

        diagram = Group(axes, plane, grid)
        diagram.add(u,pu,pxu,pyu,pxzu)       

        # dashed lines and right angles
        dxy = DashedLine(u.get_end(),pu.get_end(),dash_length=0.1).set_flat_stroke(True).set_opacity(0.5)
        rxy = RightAngleIn3D(pu,Line(pu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dx = DashedLine(u.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rx = RightAngleIn3D(pxu,Line(pxu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dy = DashedLine(u.get_end(),pyu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        ry = RightAngleIn3D(pyu,Line(pyu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)
        dxp = DashedLine(pu.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rxp = RightAngleIn3D(pxu,Line(pxu.get_end(),pu.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)        
        dxz = DashedLine(u.get_end(),pxzu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rxz = RightAngleIn3D(pxzu,Line(pxzu.get_end(),u.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)        
        dxzp = DashedLine(pxzu.get_end(),pxu.get_end(),dash_length=0.15).set_flat_stroke(True).set_opacity(0.3)
        rxzp = RightAngleIn3D(pxu,Line(pxu.get_end(),pxzu.get_end()),length=0.2).set_flat_stroke(False).set_stroke(width=1, opacity=0.5)        
        # for dash in [dxy,dx,dxp]: dash.reverse_points()
        diagram.add(dxy,rxy,dx,rx,dy,ry,dxp,rxp,dxz,rxz,dxzp,rxzp) 
        
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
        frame.scale(0.4).scale(1.25).shift(UP*0.5) # copied from awkward part elsewhere

        # add axes etc., draw vector and label it
        self.play(FadeIn(axes,plane,grid,shift=UR))
        self.play(ReplacementTransform(u.copy().scale(0.05,u.points[0]),u))        
        self.play(Write(ul))
        self.wait()

        # project to x axis   
        pxu.save_state(), dx.save_state(), rx.save_state()   
        self.play(
            TransformFromCopy(u,pxu),
            Create(dx),
            run_time=1.75,
        )
        self.play(Write(rx))
        
        # write px
        eq1 = MathTex(r"\mathbf{P_y}",r"\mathbf{P_x}","=","0",font_size=45).next_to(diagram,UP).shift(DOWN*0+RIGHT*1.5)
        eq1[0].set_color(COLOR_V2P), eq1[1].set_color(COLOR_V1P)
        self.play(Write(eq1[1]))
        self.wait() 

        # project to y, write py        
        self.play(
            pxu.animate.scale(0.01,pxu.points[0]),
            Transform(VGroup(dx,rx),Dot(axes.get_origin(),color=GREY).set_opacity(0)),
            run_time=1.75
        )
        self.remove(pxu, dx, rx)
        pxu.restore(), dx.restore(), rx.restore()
        self.play(FadeIn(eq1[0],shift=RIGHT))
        self.play(Write(eq1[2:]))
        self.wait()

        # rotate diagram
        ystuff = Group(pyu,dy,ry).set_opacity(0)        
        base = Group(axes,plane,grid,u,ul).save_state()
        self.play(
            Group(axes,plane,grid,u,ystuff).animate.rotate(-130*DEGREES,axes.z_axis.get_unit_vector()),
            ul.animate.next_to(Group(axes,plane,grid,u,pyu).copy().rotate(-130*DEGREES,axes.z_axis.get_unit_vector())[3].points[-1],LEFT,buff=0.25),
            run_time=2
        )
        self.remove(ystuff)
        pyu.set_opacity(1), dy.set_opacity(0.3), ry.set_opacity(0.5)

        # project to y axis then zero
        self.play(
            TransformFromCopy(u,pyu),
            Create(dy),
            run_time=1.75
        )
        self.play(Write(ry)) 
        self.play(
            pyu.animate.scale(0.01,pyu.points[0]),
            Transform(VGroup(dy,ry),Dot(axes.get_origin(),color=GREY).set_opacity(0)),
            run_time=1.75
        )
        self.remove(pyu)
        self.wait()

        # commuting equation
        eq2 = AlignBaseline(MathTex(r"\mathbf{P_y}",r"\mathbf{P_x}","=","0","=",r"\mathbf{P_x}",r"\mathbf{P_y}",font_size=45).move_to(eq1),eq1)
        eq2[0].set_color(COLOR_V2P), eq2[1].set_color(COLOR_V1P)
        eq2[6].set_color(COLOR_V2P), eq2[5].set_color(COLOR_V1P)
        self.play(*TransformBuilder(
            eq1,eq2,
            [
                (0,0),(1,1),(2,2),(3,3), # pypx=0
                (None,4), # =
                (1,5,TransformFromCopy,{"path_arc":-120*DEGREES}), # px
                (0,6,TransformFromCopy,{"path_arc":-120*DEGREES}), # py
            ]
            ),
            run_time=1.25
        )
        self.remove(eq1)  # symbols are being left behind for no reason?
        self.wait()

        # rotate diagram back, drop equation
        ystuff = Group(pyu,dy,ry).set_opacity(0)
        self.play(
            Restore(base),
            # Group(axes,plane,grid,u,ystuff).animate.rotate(130*DEGREES,axes.z_axis.get_unit_vector()),
            # ul.animate.next_to(Group(axes,plane,grid,u,pyu).copy().rotate(130*DEGREES,axes.z_axis.get_unit_vector())[3].points[-1],RIGHT,buff=0.35),
            FadeOut(eq2,shift=LEFT),
            run_time=2
        )
        self.remove(ystuff)
        pyu.set_opacity(1), dy.set_opacity(0.3), ry.set_opacity(0.5)

        # second case
        # write px pxy equation
        eq3 = AlignBaseline(MathTex(r"\mathbf{P_x}",r"\mathbf{P_{xy}}","=",r"\mathbf{P_x}",font_size=45).move_to(eq2),eq2)
        eq3[0].set_color(COLOR_V1P), eq3[1].set_color(COLOR_V2P), eq3[3].set_color(COLOR_V1P)
        self.play(FadeIn(eq3[1]))
        self.wait()
        self.play(FadeIn(eq3[0],shift=RIGHT))
        self.wait()

        # project to xy        
        self.play(
            TransformFromCopy(u,pu),
            Create(dxy),
            run_time=1.5,
        )
        self.play(Write(rxy))

        # project result to x
        self.play(
            TransformFromCopy(pu,pxu),
            Create(dxp),
            run_time=1.5,
        )
        self.play(Write(rxp)) 
        self.wait()

        # project straight to x again
        self.play(
            Merge([pxu,u.copy()],pxu),
            Create(dx),
            run_time=2,
        )
        self.play(Write(rx))
        self.wait()

        # show equation, = px
        self.play(FadeIn(eq3[2],shift=LEFT),run_time=1.25)
        self.play(TransformFromCopy(eq3[0],eq3[3],path_arc=-120*DEGREES),run_time=1.5)
        self.wait()

        # indicate x
        self.play(Indicate(pxu))

        # rest of equation
        eq4 = AlignBaseline(MathTex(r"\mathbf{P_x}",r"\mathbf{P_{xy}}","=",r"\mathbf{P_x}","=",r"\mathbf{P_{xy}}",r"\mathbf{P_x}",font_size=45).move_to(eq3),eq3)
        eq4[0].set_color(COLOR_V1P), eq4[1].set_color(COLOR_V2P), eq4[3].set_color(COLOR_V1P)
        eq4[6].set_color(COLOR_V1P), eq4[5].set_color(COLOR_V2P)
        self.play(*TransformBuilder(
            eq3,eq4,
            [
                (slice(0,4),slice(0,4)), # pxpxy=px
                (None,4), # =
                (1,5,TransformFromCopy,{"path_arc":-120*DEGREES}), # pxy
                (0,6,TransformFromCopy,{"path_arc":-120*DEGREES}), # px
            ]
        ), run_time=1.5)
        self.wait()

        # clear to blank diagram
        self.play(FadeOut(eq4, dx,rx,pxu,dxp,rxp,pu,dxy,rxy))
        self.wait()

        # third case
        # rotate diagram
        vectors = Group(pu,pxu,pxzu).set_opacity(0)        
        dashes = Group(dxy,dxp,dxz,dxzp).set_opacity(0)
        ras = Group(rxy,rxp,rxz,rxzp).set_opacity(0)
        base = Group(axes,plane,grid,u,ul).save_state()
        self.play(
            Group(axes,plane,grid,u,vectors,dashes,ras).animate.rotate(-60*DEGREES,axes.z_axis.get_unit_vector()),
            ul.animate.next_to(Group(axes,plane,grid,u,pyu).copy().rotate(-60*DEGREES,axes.z_axis.get_unit_vector())[3].points[-1],RIGHT,buff=0.25),
            run_time=2
        )
        self.remove(vectors, dashes,ras)
        vectors.set_opacity(1), dashes.set_opacity(0.3), 
        for ra in ras: ra.set_stroke(opacity=0.5)

        # write pxy and pxz from equation, and draw planes
        eq5 = MathTex(r"\mathbf{P_{xz}}",r"\mathbf{P_{xy}}","=",r"\mathbf{P_x}",font_size=45).next_to(diagram,UP).shift(DOWN*0+RIGHT*1.5)
        eq5[0].set_color(COLOR_V3P), eq5[1].set_color(COLOR_V2P), eq5[3].set_color(COLOR_V1P)
        xyp = OpenGLSurface(lambda u,v:axes.c2p(*[u,v,0.001]),u_range=[-0.5,1],v_range=[-0.5,1],color=COLOR_V2P).set_opacity(0.3)
        xzp = OpenGLSurface(lambda u,v:axes.c2p(*[u,0,v]),u_range=[-0.5,1],v_range=[-0.5,1],color=COLOR_V3P).set_opacity(0.3)
        self.play(FadeIn(eq5[1]))
        self.play(FadeIn(xyp,shift=DOWN),run_time=1.25)        
        self.wait()
        self.play(FadeIn(eq5[0],shift=RIGHT))
        self.play(FadeIn(xzp,shift=RIGHT),run_time=1.25)
        self.wait()

        # indicate x axis
        self.play(Indicate(axes.axes[0]))
        self.wait()
        
        # project to xy        
        self.play(
            TransformFromCopy(u,pu),
            Create(dxy),
            run_time=1.5,
        )
        self.play(Write(rxy))

        # project result to xz
        # project result to x
        pxu2 = pxu.copy().set_color(COLOR_V3P)
        self.play(
            TransformFromCopy(pu,pxu2),
            Create(dxp),
            run_time=1.5,
        )
        self.play(Write(rxp)) 
        self.wait()

        # project original to xz
        self.play(
            TransformFromCopy(u,pxzu),
            Create(dxz),
            run_time=1.5,
        )
        self.play(Write(rxz)) 
        self.wait()


        # project result to xy (merge with prior)
        self.play(
            Merge([pxu2,pxzu.copy()],pxu2.copy().set_color(COLOR_V2P)),
            Create(dxzp),
            run_time=2,
        )
        self.play(Write(rxzp))
        self.wait()

        # equation commutes
        eq6 = AlignBaseline(MathTex(r"\mathbf{P_{xz}}",r"\mathbf{P_{xy}}","=",r"\mathbf{P_x}","=",r"\mathbf{P_{xy}}",r"\mathbf{P_{xz}}",font_size=45).move_to(eq5),eq5)
        eq6[0].set_color(COLOR_V3P), eq6[1].set_color(COLOR_V2P), eq6[3].set_color(COLOR_V1P)
        eq6[6].set_color(COLOR_V3P), eq6[5].set_color(COLOR_V2P)
        self.play(*TransformBuilder(
            eq5,eq6,
            [
                (slice(0,2),slice(0,2)), # pxz pxy=px
                (None,2), # =
                (slice(0,2),3,TransformFromCopy,{"path_arc":-120*DEGREES}), # merge pxz and pxy in px
                (None,4), # =
                (1,5,TransformFromCopy,{"path_arc":-120*DEGREES}), # pxy
                (0,6,TransformFromCopy,{"path_arc":-120*DEGREES}), # pxz
            ]
        ), run_time=2.5)
        self.wait()

        # indicate Y axis
        self.play(Indicate(axes.axes[1]))
        self.wait()

        # indicate z axis
        self.play(Indicate(axes.axes[2]))
        self.wait()





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
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
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
        pul = MathTex(r"\mathbf{P_{xy}}","u", font_size=40).next_to(pu.get_end(),UR,buff=0.05).shift(RIGHT*0.4+DOWN*0.12)
        pul[0].set_color(COLOR_V1P),pul[1].set_color(COLOR_V1)
        pxul = MathTex(r"\mathbf{P_{x}}","u", font_size=40).next_to(pxu.get_end(),UP,buff=0.05).shift(LEFT*0.35)        
        pyul = MathTex(r"\mathbf{P_{y}}","u", font_size=40).next_to(pyu.get_end(),RIGHT,buff=0.05).shift(UP*0.2+LEFT*0.15)    
        pzul = MathTex(r"\mathbf{P_{z}}","u", font_size=40).next_to(pzu.get_end(),RIGHT,buff=0.1)        
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
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
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
            *[Indicate(m.get_entries()[i],scale_factor=1.6) for i in [0,4,8]],
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

        # merge so there's only one expression
        text2 = text1.copy().move_to(trse3)
        self.play(
            Merge([text1,trse3],text2)
        )
        self.wait()

        # down to the title
        self.play(FadeOut(text2))
        self.wait()



class ProjectionPropsRecap(MovingCameraScene):
    def construct(self):
        # write title
        title = Tex("Projection Matrix Properties", font_size=75).to_edge(UP)
        ul = Line(title.get_corner(DL)+DL*0.2, title.get_corner(DR)+DR*0.2, color=COLOR_V1)        
        self.add(title, ul)
        self.wait()

        # properties
        props = VGroup(*[Tex(message, font_size=55).next_to(ul, DOWN) for message in [
            r"Idempotent: $\mathbf{P}^2=\mathbf{P}$",
            r"Symmetric: $\mathbf{P}^T=\mathbf{P}$",
            r"Less-than-full rank: $\text{rank}(\mathbf{P})<n$",
            r"Eigenvalues are 0's and 1's",
            r"$\mathbf{P_2}\mathbf{P_1}=\mathbf{P_1}\mathbf{P_2}$ if orthogonal-overlapping",
            r"Orthogonal \& Non-overlapping projections add"
        ]])                
        for i in range(1,len(props)): AlignBaseline(props[i].align_to(props[0], LEFT), props[0]).shift(DOWN*i)
        props.next_to(ul,DOWN)
        for prop in props:
            color_tex(prop,(r"$\mathbf{P_1}$",COLOR_V1P),(r"$\mathbf{P_2}$",COLOR_V2P),("0",RED),("1",COLOR_V1))
            self.play(Write(prop))
            self.wait()
        
        self.wait()

        # remove props
        self.play(FadeOut(props),run_time=2.5)
        self.wait(0.5)

        # add graphic
        l1coords = np.array([2,0.5])*2        
        l2coords = np.array([0.5,2])*1.9                
        ucoords = np.array([1.5,1.5])*2
        p1coords = l1coords * np.dot(l1coords,ucoords) / np.dot(l1coords,l1coords)
        p12coords = l2coords * np.dot(l2coords,p1coords) / np.dot(l2coords,l2coords)
        p2coords = l2coords * np.dot(l2coords,ucoords) / np.dot(l2coords,l2coords)
        p21coords = l1coords * np.dot(l1coords,p2coords) / np.dot(l1coords,l1coords)
        
        # initial frame stuff
        frame = self.camera.frame        

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

        diagram.shift(DOWN*3.3+LEFT*2.7).scale(1.55)
        diagram.remove(axes)
        self.play(FadeIn(diagram),run_time=2.5)
        self.wait(2)



class Outro(Scene):
    def construct(self):
        # credits
        author = Tex("Created by Sam Levey", font_size=80).to_edge(UP)
        self.play(FadeIn(author, shift=RIGHT), run_time=1.25)

        thanks = Tex("Special Thanks to Ben Golub", font_size=55).shift(UP*0.5)   

        # music credit
        music = Tex("Music by Karl Casey @ White Bat Audio",font_size=55).next_to(thanks,DOWN)
        self.play(FadeIn(thanks,shift=DOWN))
        self.play(FadeIn(music,shift=DOWN))

        # thanks
        # thanks = Tex(r"Special thanks to: \\ Josh Perlman \\ Conner Howell", font_size=65)
        # self.play(FadeIn(thanks, shift=RIGHT))

        # Banner animation
        banner = ManimBanner()
        banner.scale(0.3)
        banner.to_edge(DOWN)
        banner.shift(RIGHT*2)
        self.play(FadeIn(banner))
        made_with = Tex("Made with ")
        made_with.scale(1.5)
        made_with.next_to(banner, LEFT, buff=1.2)
        made_with.align_to(banner.M, DOWN)
        url = Tex("\\verb|https://manim.community|")
        url.next_to(VGroup(made_with, banner), DOWN, buff=-0.2)
        url.align_to(made_with, LEFT)
        self.play(AnimationGroup(
            AnimationGroup(banner.expand(), Write(made_with)),
            FadeIn(url),
            lag_ratio=0.5
        ))
        self.wait(2)

        # Remove things

        self.play(Unwrite(author), Unwrite(VGroup(thanks,music)))
        self.play(Uncreate(banner), Unwrite(made_with), Unwrite(url))
        self.wait()
        

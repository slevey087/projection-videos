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


def color_tex_standard(equation):
    if isinstance(equation,Matrix):
        for entry in equation.get_entries(): color_tex_standard(entry)
        return equation
    return color_tex(equation,(r"\mathbf{v}", VCOLOR), (r"\mathbf{x}",XCOLOR),(r"\mathbf{y}",YCOLOR), (r"\mathbf{p}",PCOLOR),("p_x",PCOLOR),("p_y",PCOLOR), (r"\mathbf{u}", UCOLOR),(r"\hat{\mathbf{u}}", PUCOLOR),(r"\hat{\mathbf{v}}", PVCOLOR))



class test(ThreeDScene):
    def construct(self):
        xcoords = np.array([1,0,0])
        ycoords = np.array([0,1,0])
        vcoords = np.array([0.5,0.7,0.7])
        pxcoord = np.dot(xcoords,vcoords)
        pycoord = np.dot(ycoords,vcoords)
        pcoords = pxcoord*xcoords + pycoord*ycoords        

        # define diagram
        axes = ThreeDAxes(
            x_range=[0,0.5],x_length=2.5 / 2,
            y_range=[0,0.5],y_length=2.5 / 2,
            z_range=[0,0.5],z_length=2.5 / 2,
        ).set_opacity(0)        
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR)        
        x = Arrow(axes @ ORIGIN, axes @ xcoords, buff=0, color=XCOLOR)
        y = Arrow(axes @ ORIGIN, axes @ ycoords, buff=0, color=YCOLOR)
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR)        
        px = Arrow(axes @ ORIGIN, axes @ (pxcoord*xcoords), buff=0,color=PCOLOR)
        py = Arrow(axes @ ORIGIN, axes @ (pycoord*ycoords), buff=0,color=PCOLOR)
        dy = DashedLine(axes @ pcoords, axes @ (pxcoord*xcoords), dash_length=0.15).set_opacity(0.4)
        dx = DashedLine(axes @ pcoords, axes @ (pycoord*ycoords), dash_length=0.15).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
        rx = Arrow(axes @ xcoords, axes @ (xcoords + vcoords-pcoords), buff=0, color=RCOLOR)        
        ArrowGradient(rx,[PCOLOR,VCOLOR])        
        ry = Arrow(axes @ ycoords, axes @ (ycoords + vcoords-pcoords), buff=0, color=RCOLOR)        
        ArrowGradient(ry,[PCOLOR,VCOLOR])        
        ra = VGroup(
            Line(axes @ (0.9*pcoords),axes @ (0.9*pcoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*pcoords+OUT*0.1),axes @ (1*pcoords+OUT*0.1), stroke_width=2)
        )
        rax = VGroup(
            Line(axes @ (0.9*xcoords),axes @ (0.9*xcoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*xcoords+OUT*0.1),axes @ (1*xcoords+OUT*0.1), stroke_width=2)
        )
        ray = VGroup(
            Line(axes @ (0.9*ycoords),axes @ (0.9*ycoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*ycoords+OUT*0.1),axes @ (1*ycoords+OUT*0.1), stroke_width=2)
        )
        vectors = VGroup(v,x,y,p,px,py,r,rx,ry)
        dashes = VGroup(dy,dx)        

        plane = Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=1).set_opacity(0.5)
        
        diagram = VGroup(axes,plane,vectors,dashes, ra,rax,ray)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UP,buff=0.25)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UR,buff=0.15).shift(LEFT*0.13)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl)
        diagram.add(labels)        
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.5+RIGHT*0.2)
        self.set_camera_orientation(frame_center=IN*11) # self.set_camera_orientation(zoom=2)
        for vector in vectors: 
            ArrowStrokeFor3dScene(self,vector,family=True)        
        
        self.add(diagram)        
        self.set_camera_orientation(frame_center=ORIGIN)
        diagram.to_corner(UR)
        self.remove(rx,ry,rax,ray)

        nex3 = MathTex(r"p_x \mathbf{x} \cdot \mathbf{x} + p_y \mathbf{x} \cdot \mathbf{y}","=",r"\mathbf{x} \cdot \mathbf{v}", font_size=65).shift(UP*2+LEFT*3)
        ney3 = MathTex(r"p_x \mathbf{y} \cdot \mathbf{x} + p_y \mathbf{y} \cdot \mathbf{y}","=",r"\mathbf{y} \cdot \mathbf{v}", font_size=65).shift(DOWN*2+LEFT*3)   
        c1 = MathTex(r"\text{Case 1: }", r"\mathbf{x}, \mathbf{y} \text{ Orthonormal}")        
        self.add(nex3,ney3,c1)        
        self.wait()

        # zoom back in to diagram
        for vector in vectors: vector.add_updater(ArrowStrokeCameraUpdater(self))
        self.move_camera(
            frame_center=VGroup(v,p,r).get_center()+DOWN*0.5+LEFT*0.2+IN*11,
            added_anims=[
                VGroup(r,rl,ra,px,py,dx,dy,pxl,pyl,p,pl).animate.set_opacity(0)
            ],
            run_time=2
        )           
        for mob in [rx,ry,rax,ray]: mob.set_opacity(0)
        for vector in vectors: vector.clear_updaters()        
        self.wait()

        # project x        
        self.play(Rotate(diagram,-55*DEGREES,axis=(axes@(0,0,1))-(axes @ ORIGIN)),about_point=axes@(0.5,0.45,0),run_time=2)
        dpx = DashedLine(v.get_end(),px.get_end(),dash_length=0.15).set_opacity(0.4)      
        rapx = VGroup(
            Line(axes @ (0.9*pxcoord*xcoords),(axes @ (0.9*pxcoord*xcoords))+dpx.get_unit_vector()*-0.2, stroke_width=2),
            Line((axes @ (0.9*pxcoord*xcoords))+dpx.get_unit_vector()*-0.2,(axes @ (1*pxcoord*xcoords))+dpx.get_unit_vector()*-0.2, stroke_width=2)
        )        
        self.remove(px)
        px.set_opacity(1)
        self.play(
            TransformFromCopy(v,px),
            Write(dpx),
            run_time=2
        )
        self.play(Write(rapx))
        diagram.add(dpx,rapx)
        
        # project y
        self.play(Rotate(diagram,90*DEGREES,axis=(axes@(0,0,1))-(axes @ ORIGIN)),about_point=axes@(0.5,0.45,0),run_time=2)
        dpy = DashedLine(v.get_end(),py.get_end(),dash_length=0.15).set_opacity(0.4)      
        rapy = VGroup(
            Line(axes @ (0.9*pycoord*ycoords),(axes @ (0.9*pycoord*ycoords))+dpy.get_unit_vector()*-0.2, stroke_width=2),
            Line((axes @ (0.9*pycoord*ycoords))+dpy.get_unit_vector()*-0.2,(axes @ (1*pycoord*ycoords))+dpy.get_unit_vector()*-0.2, stroke_width=2)
        )        
        self.remove(py)
        py.set_opacity(1)
        self.play(
            TransformFromCopy(v,py),
            Write(dpy),
            run_time=2
        )
        self.play(Write(rapy))        
        diagram.add(dpy,rapy)

        # restore diagram angle
        self.play(Rotate(diagram,-35*DEGREES,axis=(axes@(0,0,1))-(axes @ ORIGIN)),about_point=axes@(0.5,0.45,0),run_time=2)
        self.wait()

        # vector sum
        py.save_state()
        py.add_updater(ArrowStrokeCameraUpdater(self))
        self.play(
            py.animate.put_start_and_end_on(px.get_end(),p.get_end()),
            run_time=1.25
        )
        self.remove(p)
        p.set_opacity(1)
        self.play(GrowArrow(p))
        self.play(Restore(py),run_time=1.25)
        py.clear_updaters()
        self.wait()

        # zoom back out and restore diagram
        for vector in vectors: vector.add_updater(ArrowStrokeCameraUpdater(self))
        for mob in [rx,ry,rax,ray]: mob.set_opacity(0)
        self.move_camera(
            frame_center=ORIGIN,
            added_anims=[
                FadeOut(dpx,rapx,dpy,rapy),
                VGroup(r,rl,ra,p,pl,pxl,pyl).animate.set_opacity(1),
                VGroup(dx,dy).animate.set_opacity(0.4)
            ],
            run_time=2
        )
        diagram.remove(dpx,rapx,dpy,rapy)
        for vector in vectors: vector.clear_updaters()
        for mob in [rx,ry,rax,ray]: self.remove(mob.set_opacity(1))
        self.wait()

# config.from_animation_number = 2
# config.upto_animation_number = 3

class test1(ThreeDScene):
    def construct(self):
        # Arrow.set_default(flat_stroke = False)

        xcoords = np.array([0.7,0.35,0])
        ycoords = np.array([0.3,1.1,0])
        vcoords = np.array([0.5,0.7,0.7])
        amatrix = np.vstack([xcoords,ycoords]).T
        pxcoord = np.matmul(np.linalg.inv(np.matmul(amatrix.T,amatrix)),np.matmul(amatrix.T, vcoords))[0]
        pycoord = np.matmul(np.linalg.inv(np.matmul(amatrix.T,amatrix)),np.matmul(amatrix.T, vcoords))[1]
        pcoords = pxcoord*xcoords + pycoord*ycoords

        # define diagram
        axes = ThreeDAxes(
            x_range=[0,0.5],x_length=2.5 / 2,
            y_range=[0,0.5],y_length=2.5 / 2,
            z_range=[0,0.5],z_length=2.5 / 2,
        ).set_opacity(0)        
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR)        
        x = Arrow(axes @ ORIGIN, axes @ xcoords, buff=0, color=XCOLOR)
        xp = Arrow(axes @ ORIGIN, axes @ (1,0,0), buff=0, color=XCOLOR)
        y = Arrow(axes @ ORIGIN, axes @ ycoords, buff=0, color=YCOLOR)        
        yp = Arrow(axes @ ORIGIN, axes @ (0,1,0), buff=0, color=YCOLOR)
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR)        
        px = Arrow(axes @ ORIGIN, axes @ (pxcoord*xcoords), buff=0,color=PCOLOR)
        py = Arrow(axes @ ORIGIN, axes @ (pycoord*ycoords), buff=0,color=PCOLOR)
        dy = DashedLine(axes @ pcoords, axes @ (pxcoord*xcoords), dash_length=0.15).set_opacity(0.4)
        dx = DashedLine(axes @ pcoords, axes @ (pycoord*ycoords), dash_length=0.15).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR)        
        # ArrowGradient(r,[PCOLOR,VCOLOR])
        ra = VGroup(
            Line(axes @ (0.9*pcoords),axes @ (0.9*pcoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*pcoords+OUT*0.1),axes @ (1*pcoords+OUT*0.1), stroke_width=2)
        )
        vectors = VGroup(v,x,y,p,px,py,r,xp,yp)
        dashes = VGroup(dy,dx)

        plane = Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=1).set_opacity(0.5)
        
        diagram = VGroup(axes,plane,vectors,dashes, ra)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        xpl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(xp.get_end(),LEFT,buff=0.15)
        ypl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(yp.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UP,buff=0.25)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UR,buff=0.15)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl,xpl,ypl)
        diagram.add(labels)        
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.5+RIGHT*0.2).shift(DOWN*0.1)
        
        # self.set_camera_orientation(zoom=2.5)
        # for vector in vectors: vector.set_flat_stroke(False)

        
        self.add(diagram)
        self.remove(r,rl,xp,yp,xpl,ypl)
        self.play(diagram.animate.rotate(60*DEGREES,axis = RIGHT).rotate(45*DEGREES,axis=(axes @ (0,0,1))-(axes @ ORIGIN)).shift(DOWN*0.25))
        for vector in vectors: 
            face_camera(self, vector)
            vector.add_updater(ArrowStrokeCameraUpdater(self))
        
        self.move_camera(frame_center=IN*15,run_time=3)
        for vector in vectors: vector.clear_updaters()
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

        xlabel = MathTex("x",font_size=15,color=BLACK).set_opacity(0.3).move_to(axes.get_axes()[0].get_tip())
        ylabel = MathTex("y",font_size=12,color=BLACK).set_opacity(0.3).move_to(axes.get_axes()[1].get_tip())
        zlabel = MathTex("z",font_size=15,color=BLACK).set_opacity(0.3).move_to(axes.get_axes()[2].get_tip())
        diagram.add(xlabel,ylabel,zlabel)

        self.add(diagram)
        



class scratch2(MovingCameraScene):
    def construct(self):        
        xcoords = np.array([1,0,0])
        ycoords = np.array([0,1,0])
        vcoords = np.array([0.5,0.7,0.7])
        pxcoord = np.dot(xcoords,vcoords)
        pycoord = np.dot(ycoords,vcoords)
        pcoords = pxcoord*xcoords + pycoord*ycoords

        # set up frame
        frame = self.camera.frame
        frame.save_state()

        # define diagram
        axes = ThreeDAxes(
            x_range=[-2,2],x_length=10,
            y_range=[-2,2],y_length=10,
            z_range=[-2,2],z_length=10,
        )                
        v = Arrow(axes.c2p(*ORIGIN), axes.c2p(*vcoords), buff=0, color=VCOLOR)        
        x = Arrow(axes.c2p(*ORIGIN), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        y = Arrow(axes.c2p(*ORIGIN), axes.c2p(*ycoords), buff=0, color=YCOLOR)
        p = Arrow(axes.c2p(*ORIGIN), axes.c2p(*pcoords), buff=0, color=PCOLOR)        
        px = Arrow(axes.c2p(*ORIGIN), axes.c2p(*(pxcoord*xcoords)), buff=0,color=PCOLOR)
        py = Arrow(axes.c2p(*ORIGIN), axes.c2p(*(pycoord*ycoords)), buff=0,color=PCOLOR)
        dy = DashedLine(axes.c2p(*pcoords), axes.c2p(*(pxcoord*xcoords)), dash_length=0.15)
        dx = DashedLine(axes.c2p(*pcoords), axes.c2p(*(pycoord*ycoords)), dash_length=0.15)
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
        r.get_tip().set_stroke(RED)
        ra = VGroup(
            Line(axes.c2p(*(0.9*pcoords)),axes.c2p(*(0.9*pcoords+OUT*0.1))),
            Line(axes.c2p(*(0.9*pcoords+OUT*0.1)),axes.c2p(*(1*pcoords+OUT*0.1)))
        )
        plane = Surface(lambda u,v:[u,v,0],u_range=[-0.75,3],v_range=[-0.75,3],resolution=1).set_opacity(0.5)
        diagram = Group(plane,x,y,v,p,px,py,r,ra,dy,dx)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)
        for arrow in [v,x,y,p,px,py,r]: face_camera(self,arrow)
        diagram.to_corner(UR).shift(DOWN*0.5+LEFT*0.25)
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UP,buff=0.1)
        pxl[0][:2].set_color(PCOLOR), pxl[0][-1].set_color(XCOLOR)
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UP,buff=0.1)
        pyl[0][:2].set_color(PCOLOR), pyl[0][-1].set_color(YCOLOR)
        rl = MathTex(r"\mathbf{v-p}", color=[VCOLOR,PCOLOR], font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.15).set_sheen_direction(UP)
        diagram.add(xl,vl,yl,pl,pxl,pyl,rl)
        frame.scale(0.5).move_to(diagram).shift(DOWN*0.35)
        
        self.add(diagram)
        
        
        # zoom to x, then lengthen x
        # frame.move_to(r.get_end()).scale(0.3)
        
        

        """
        self.play(
            x.animate.scale(1.2,about_point=x.points[0]),
            xl.animate.shift(0.3*(x.points[-1]-x.points[0])/np.linalg.norm(x.points[-1]-x.points[0]))
        )"""


        """
        # zoom to y then lengthen y
        self.play(frame.animate.move_to(y.points[-1]).scale(0.8),run_time=2)
        self.play(
            y.animate.scale(0.85,about_point=y.points[0]),
            yl.animate.shift(-0.2*(y.points[-1]-y.points[0])/np.linalg.norm(y.points[-1]-y.points[0]))
        )
        self.wait(w)

        # zoom back out
        self.play(Restore(frame),run_time=2.5)
        self.wait(w)

"""




class SymmetricGeom2d(ThreeDScene):
    def construct(self):        
        h = 0.6
        ucoords = np.array([0.7,0.25,h])
        pucoords = ucoords - np.array([0,0,h])        
        vcoords = np.array([0.25,0.7,h])
        pvcoords = vcoords - np.array([0,0,h])

        # set up frame
        class FrameClass():
            def __init__(self,camera):
                self.camera = camera
            
            def scale(self,z):
                self.camera.set_zoom(self.camera.get_zoom()/z)
                return self
            
            def move_to(self,p):
                self.camera.frame_center=p
                return self
            
            def shift(self,d):
                self.camera.frame_center = self.camera.frame_center + d
                return self
            
            def save_state(self):
                pass

        frame = FrameClass(self.camera)
        

        # define diagram
        axes = ThreeDAxes(
            x_range=[-2,2],x_length=10,
            y_range=[-2,2],y_length=10,
            z_range=[-2,2],z_length=10,
        )                
        u = Arrow(axes.c2p(*ORIGIN), axes.c2p(*ucoords), buff=0, color=COLOR_V1)        
        v = Arrow(axes.c2p(*ORIGIN), axes.c2p(*vcoords), buff=0, color=COLOR_V2)                
        pu = Arrow(axes.c2p(*ORIGIN), axes.c2p(*pucoords), buff=0, color=COLOR_V1P)        
        pv = Arrow(axes.c2p(*ORIGIN), axes.c2p(*pvcoords), buff=0, color=COLOR_V2P)        
        plane = Surface(lambda u,v:[u,v,0],u_range=[-0.75,2.5],v_range=[-0.75,2.5],resolution=1).set_opacity(0.4)
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




"""
with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()
"""
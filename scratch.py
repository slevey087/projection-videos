from manim import *
from manim.opengl import *
# from scipy import stats
# import numpy as np

from utils import *

w=1


# new colors
COLOR_V1 =  XKCD.LIGHTBLUE
COLOR_V1P = XKCD.WINDOWSBLUE # pureblue and cobaltblue is good too
COLOR_V2 = XKCD.LIGHTAQUA
COLOR_V2P = XKCD.BLUEGREEN
COLOR_V3P = XKCD.ELECTRICPURPLE # XKCD.EASTERPURPLE and XKCD.LIGHTISHPURPLE are good too


config.renderer="opengl"
class test(Scene):
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
        frame.scale(0.4).scale(1.25).shift(UP*0.5) # copied from awkward part elsewh

        self.add(axes, grid,plane,u,ul)

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
        self.play(FadeIn(xzp,shift=DR),run_time=1.25)
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


        self.interactive_embed()



"""
with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = test()
    scene.render()
"""
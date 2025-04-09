from manim import *
from scipy import stats
import numpy as np

from utils import *




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

BCOLOR = XKCD.ELECTRICPURPLE

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
        (r"\hat{\mathbf{v}}", PVCOLOR),
        (r"\mathbf{b}",BCOLOR)
    )




class Oblique1D(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,0.7])
        vcoords = np.array([1,2.2])
        k = 0.55 # parameter to control degree of oblique projection. At 1, it's the orthogonal projection; at 0, it's 0.
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * k
        bcoords = 2 * np.array([1,-((vcoords - pcoords)[0]) / ((vcoords - pcoords)[1])])  # formula here is based on tha the dot product with r must be 0
        zcoords = bcoords * np.dot(vcoords, bcoords) / np.dot(bcoords,bcoords) 
        
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
        b = Arrow(axes @ ORIGIN, axes @ bcoords,buff=0, color=BCOLOR)
        vectors = VGroup(x,v,p,r,b)       

        angle = Angle(p,r,radius=0.35,quadrant=(-1,1),other_angle=True) 
        dx = DashedLine(v.get_end(),p.get_end(),dash_length=0.1).set_opacity(0.6)
        db = DashedLine(v.get_end(),axes @ zcoords,dash_length=0.1).set_opacity(0.6)
        rab = RightAngle(db, b,length=0.2,quadrant=(-1,1)).set_stroke(opacity=0.6)



        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DR,buff=0.03)        
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r.get_center()).shift(UP*0.1+RIGHT*0.2)
        color_tex_standard(rl)
        bl = MathTex(r"\mathbf{b}", font_size=60, color=BCOLOR).next_to(b.get_tip(),buff=0.15)        
        labels = VGroup(xl, vl, pl, rl, bl)
        
        diagram = VGroup(axes, vectors, labels,angle,dx,db,rab)
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
        pe = MathTex(r"\mathbf{p}","=",r"p_x \mathbf{x}","=",r"\mathbf{P}\mathbf{v}", font_size=60)        
        color_tex_standard(pe)
        pe.shift(pl[0].get_center()-pe[0].get_center())
        self.play(Write(pe[1]))
        self.play(
            Write(pe[2][:2]),
            TransformFromCopy(xl[0],pe[2][-1]),
            run_time=1.25
        )
        self.wait(w)

        # projector equation, after moving everything over
        visible = VGroup(x,v,p,xl,vl,pl,pe[1:3])
        self.play(visible.animate.shift(LEFT*1),run_time=1.5)
        pe[3:].shift(LEFT*1)
        self.play(Write(pe[3]))
        self.play(
            Write(pe[4][0]), # P
            TransformFromCopy(vl[0],pe[4][1]), # v
            run_time=1.5
        )
        self.wait()


        # remove p equation, move everything back    
        self.play(FadeOut(pe[1:]))
        visible = VGroup(x,v,xl,vl,p,pl)
        self.play(visible.animate.shift(RIGHT*1),run_time=1.5)
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

        # dot product not zero
        visible = VGroup(x,v,p,r,rl,xl,vl,pl,angle)
        self.play(visible.animate.shift(LEFT*1))
        vpdxnz = MathTex("(",r"\mathbf{v}-\mathbf{p}",")",r"\cdot", r"\mathbf{x}",r"\neq 0",font_size=60)
        color_tex_standard(vpdxnz)
        vpdxnz.shift(rl.get_center()-vpdxnz[1].get_center())
        self.play(
            FadeIn(vpdxnz[0], vpdxnz[2])
        )
        self.play(FadeIn(vpdxnz[3],shift=DOWN)) # dot
        self.play(TransformFromCopy(xl,vpdxnz[4])) # x
        self.play(Write(vpdxnz[5:])) # neq0
        self.wait()

        # remove dot product equation
        self.play(
            FadeOut(vpdxnz,shift=DR), 
            visible.animate.shift(RIGHT*1),
            run_time=1.5)

        self.wait()

        # animation where angle moves
        self.remove(r,rl,p,angle)
        k_anim = ValueTracker(k)
        p_anim = always_redraw(lambda: Arrow(axes @ ORIGIN, axes @ ((xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords))*k_anim.get_value()), buff=0, color=PCOLOR))
        r_anim = always_redraw(lambda: ArrowGradient(Arrow(p_anim.get_end(), v.get_end(), buff=0),[PCOLOR,VCOLOR]))        
        rl_anim = always_redraw(lambda: color_tex_standard(MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r_anim.get_center()).shift(UP*0.1+RIGHT*0.2)))
        angle_anim = always_redraw(lambda: Angle(p_anim,r_anim,radius=0.35,quadrant=(-1,1),other_angle=True if k_anim.get_value()>=0 else False) )
        self.add(p_anim,r_anim, rl_anim,angle_anim)
        self.play(k_anim.animate.set_value(1.6),run_time=2)
        self.play(k_anim.animate.set_value(-0.5),run_time=2)
        self.play(k_anim.animate.set_value(k),run_time=2)
        self.remove(p_anim,r_anim, rl_anim,angle_anim)
        self.add(r,rl,p,angle)
        self.wait()

        # flash rejection
        tr = VMobject().add_points_as_corners(
            [r.get_end(),r.get_start()]
        ).set_color(YELLOW)
        mask = Rectangle(height=8,width=12).set_fill(BLACK,0.6).set_stroke(BLACK).move_to(diagram)
        self.play(FadeIn(mask),run_time=1.25)
        self.play(ShowPassingFlash(tr,time_width=0.2),run_time=2)
        self.play(FadeOut(mask),run_time=1.25)
        self.wait()

        # add b vector
        self.play(
            FadeOut(r,rl, angle,pl),
            frame.animate.scale(0.75).shift(DOWN*0.1),
            run_time=1.5)
        self.play(
            LaggedStart(
                Write(db),
                GrowArrow(b), 
                lag_ratio=0.5
            ),                                   
            run_time=2.5
        )
        self.wait()

        # b label
        self.play(Write(bl))
        self.wait()

        # zoom in for right angle
        self.play(frame.animate.move_to(rab).scale(0.3/0.75),run_time=2)
        self.play(Write(rab),run_time=1.25)
        self.wait()

        # zoom back to diagram to add stuff back in
        self.play(frame.animate.move_to(diagram).scale(1/0.3),run_time=2)
        self.play(FadeIn(r,rl,pl.move_to(angle).shift(UL*0.1)))
        self.wait()

        # restore frame
        self.play(Restore(frame), run_time=2)
        self.wait()

        # normal equation
        ne = MathTex(r"\mathbf{b} \cdot (\mathbf{v}-\mathbf{p})","=","0", font_size=70).shift(LEFT*2)
        color_tex_standard(ne)
        self.play(TransformFromCopy(bl,ne[0][0]), run_time=1.75)
        self.play(Write(ne[0][1]))
        self.play(
            Write(ne[0][2]), Write(ne[0][-1]), # parentheses
            TransformFromCopy(rl[:],ne[0][3:6])
        , run_time=1.75)
        self.play(Write(ne[1]))
        self.play(Write(ne[2]))
        self.wait(w)

        # distribute
        ne2 = MathTex(r"\mathbf{b}\cdot \mathbf{v} - \mathbf{b} \cdot \mathbf{p} = 0", font_size=70).shift(LEFT*2)
        color_tex_standard(ne2)
        self.play(*TransformBuilder(
            ne, ne2,
            [
                ([0,0],[0,0]), # x
                ([0,0],[0,4], TransformFromCopy,{"path_arc":120*DEGREES}), # 2nd x
                ([0,1], [0,1]), # dot
                ([0,1],[0,5], TransformFromCopy,{"path_arc":120*DEGREES}), # 2nd dot
                ([0,[2,6]],None), # parentheses
                ([0,3],[0,2]), # v
                ([0,4], [0,3]), # -
                ([0,5],[0,6]), # p
                ([1,0], [0,-2]), # =
                ([2,0], [0,-1]), #0
            ]
        ), run_time=1.5)
        self.wait(w)

        # add across
        ne3 = MathTex(r"\mathbf{b}\cdot \mathbf{v} = \mathbf{b} \cdot \mathbf{p}", font_size=70).shift(LEFT*2)
        color_tex_standard(ne3)
        self.play(*TransformBuilder(
            ne2,ne3,
            [
                ([0,[0,1,2]],[0,[0,1,2]]), # x dot v
                ([0,-2], [0,3]), # =
                ([0,[4,5,6]], [0,[4,5,6]],None,{"path_arc":-190*DEGREES}), # x dot p
                ([0,3], None), # -
                ([0,-1], None), # 0
            ]
        ),run_time=1.5)
        self.wait(w)

        # flip sides
        ne4 = MathTex(r"\mathbf{b}\cdot \mathbf{p} = \mathbf{b} \cdot \mathbf{v}", font_size=70).shift(LEFT*2)
        color_tex_standard(ne4)
        self.play(*TransformBuilder(
            ne3,ne4,
            [
                ([0,[0,1,2]],[0,[4,5,6]],None,{"path_arc":-160*DEGREES}), #x dot v
                ([0,3],[0,3]), # =
                ([0,[4,5,6]], [0,[0,1,2]],None,{"path_arc":-160*DEGREES}), # x dot p
            ]
        ), run_time=1.5)
        self.wait(w)

        # write p-x formula
        pfe = MathTex(r"\mathbf{p} =", r"\frac{\mathbf{b} \cdot \mathbf{v}}{\mathbf{b}\cdot \mathbf{x}}", r"\mathbf{x}", font_size=70).next_to(ne4,UP)
        pfe.shift(RIGHT*(ne4[0][2].get_center()[0]-pfe[0][1].get_center()[0]))
        color_tex_standard(pfe)
        pfe0 = MathTex(r"\mathbf{p} =", "p_x", r"\mathbf{x}", font_size=70).move_to(pfe,aligned_edge=LEFT)
        color_tex_standard(pfe0)
        self.play(
            FadeIn(pfe0,shift=LEFT),
            ne4.animate.shift(DOWN)
        )
        self.wait()

        # substitute in
        ne5 = MathTex(r"\mathbf{b}\cdot p_x \mathbf{x} = \mathbf{b} \cdot \mathbf{v}", font_size=70).shift(LEFT*2+DOWN)
        color_tex_standard(ne5)
        self.play(*TransformBuilder(
            [ne4[0],pfe0],ne5,
            [
                ([0,[0,1]],[0,[0,1]]), # x dot
                ([0,2],None), # p
                ([1,1,slice(None,None)],[0,slice(2,4)],TransformFromCopy), # px
                ([1,2,0],[0,4], TransformFromCopy), # x
                ([0,slice(3,None)],[0,slice(5,None)]), # = onward
            ]
        ), run_time=1.5)
        self.wait(w)

        # solve for px
        ne6 = MathTex(r"p_x = \frac{\mathbf{b} \cdot \mathbf{v}}{\mathbf{b} \cdot \mathbf{x}}", font_size=70).shift(LEFT*2+DOWN)
        color_tex_standard(ne6)
        ne6.shift(RIGHT * (pfe[0][1].get_center()[0] - ne6[0][2].get_center()[0]))
        self.play(*TransformBuilder(
            ne5[0],ne6[0],
            [
                (0,7,None,{"path_arc":120*DEGREES}), # x
                (1,8,None,{"path_arc":120*DEGREES}), # dot
                ([[2,3]],[[0,1]]), # px
                (4,9,None,{"path_arc":120*DEGREES}), # x
                (5,2), # =
                (slice(6,9), slice(3,6)), # x dot v
                (None, 6)
            ]
        ), run_time=1.5)
        self.wait(w)

        # full equation for p    
        # rename ne6 to coef, to match the code I'm copying this from from part 2
        coef = ne6    

        # subsitute
        self.play(
            ReplacementTransform(pfe0[0],pfe[0]),
            ReplacementTransform(pfe0[-1],pfe[-1]),
            # Merge([coef[1].copy(),pfe0[1]], pfe[1])
            FadeOut(pfe0[1]), TransformFromCopy(coef[0][3:],pfe[1][:]),
            run_time=1.75
        )
        self.wait(w)

        # shift x around
        pfe1 = MathTex(r"\mathbf{p} =", r"\mathbf{x}",r"\frac{\mathbf{b} \cdot \mathbf{v}}{\mathbf{b}\cdot \mathbf{x}}",  font_size=70)
        pfe1.shift(pfe[0][1].get_center() - pfe1[0][1].get_center())
        color_tex_standard(pfe1)
        self.play(*TransformBuilder(
            pfe,pfe1,
            [
                (0,0), # p=
                (2,1,None,{"path_arc":260*DEGREES}), # x
                (1,2), # bv/bx
            ]
        ), run_time=1.5)
        self.wait()

        # to matrix
        pfe2 = MathTex(r"\mathbf{p} =", r"\mathbf{x}",r"\frac{\mathbf{b}^T \mathbf{v}}{\mathbf{b}^T \mathbf{x}}",  font_size=70)
        pfe2.shift(pfe[0][1].get_center() - pfe2[0][1].get_center())
        color_tex_standard(pfe2)  
        coef2 = MathTex(r"p_x = \frac{\mathbf{b}^T \mathbf{v}}{\mathbf{b}^T \mathbf{x}}", font_size=70).shift(LEFT*2+DOWN)
        color_tex_standard(coef2)
        coef2.shift(coef[0][2].get_center()-coef2[0][2].get_center())
        self.play(
            *TransformBuilder(
                pfe1,pfe2,
                [
                    (0,0), # p=
                    (1,1), # x
                    ([2,0],[2,0]), # b
                    ([2,1],[2,1]), # dot to T
                    ([2,slice(2,5)],[2,slice(2,5)]), # v frac b
                    ([2,5],[2,5]), # dot to T
                    ([2,6],[2,6]) # x
                ]
            ),
            *TransformBuilder(
                coef,coef2,
                [
                    ([0,slice(0,4)],[0,slice(0,4)]), # px = b
                    ([0,4],[0,4]), # dot to T
                    ([0,slice(5,8)],[0,slice(5,8)]), #v frac b
                    ([0,8],[0,8]), # dot to T
                    ([0,9],[0,9])
                ]
            ),
            run_time=1.5
        )
        self.wait()

        # pull denominators out
        pfe3 = MathTex(r"\mathbf{p} =", r"\mathbf{x}",r"\left(\mathbf{b}^T \mathbf{x} \right)^{-1}",r"\mathbf{b}^T \mathbf{v}",  font_size=70)
        AlignBaseline(pfe3.move_to(pfe2),pfe2)
        color_tex_standard(pfe3)  
        self.play(
            *TransformBuilder(
                pfe2,pfe3,
                [
                    (0,0), # p=
                    (1,1), # x
                    (None,[2,[0,4]],FadeIn), # ()
                    ([2,[4,5,6]],[2,[1,2,3]],None,{"path_arc":-120*DEGREES}), # btx
                    ([2,3],None), # frac
                    (None,[2,[5,6]],FadeIn), # -1
                    ([2,[0,1,2]], [3,[0,1,2]],None,{"path_arc":-120*DEGREES}), # btv

                ]
            ),
            run_time=2
        )
        self.wait()
        coef3 = MathTex(r"p_x = \left(\mathbf{b}^T \mathbf{x} \right)^{-1} \mathbf{b}^T \mathbf{v}", font_size=70).shift(LEFT*2+DOWN)
        color_tex_standard(coef3)
        AlignBaseline(coef3.move_to(coef2),coef2)
        self.play(
            *TransformBuilder(
                coef2,coef3,
                [
                    ([0,slice(0,3)],[0,slice(0,3)]), # px=
                    (None,[0,[3,7]],FadeIn), # ()
                    ([0,slice(7,10)],[0,slice(4,7)],None,{"path_arc":-120*DEGREES}), # btx
                    (None,[0,slice(8,10)],FadeIn), #-1
                    ([0,slice(3,6)],[0,slice(10,None)],None,{"path_arc":-120*DEGREES}), # btv
                    ([0,6],None) # frac
                ]
            ),run_time=2
        )
        self.wait()









config.from_animation_number = 83
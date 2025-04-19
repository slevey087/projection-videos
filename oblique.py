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
        (r"\mathbf{b}",BCOLOR),
        (r"\mathbf{b_1}",BCOLOR),
        (r"\mathbf{b_2}",BCOLOR),
        (r"\mathbf{x_1}", XCOLOR),
        (r"\mathbf{x_2}", XCOLOR),
        (r"p_1}", PCOLOR),
        (r"p_2}", PCOLOR),
    )




def alter_diagram(scene,animation_func,hidden_mobs=[],run_time=1,move_camera_dict=None,updater=None,updater_mobs=[],added_anims=[],opacities={}):
    """
        Hacky workaround to deal with the fact that you can't animate groups without all the group items appearing on screen.

        scene: the scene
        animation_func: a function (like a lambda) that will return the animation you want to run. Needs to be this way because animation has to be created after other mobs are hidden.
        hidden_mobs: list of mobs which should stay invisible, despite being technically in the animation
        run_time
        move_camera_dict: dictionary if you want to change 3d scene parameters (like frame_center, phi, etc.)
        updater: attach an updater, optional
        updater_mobs: which mobs to attach the updater to
        added_anims: additional animations (should not contain the hidden mobs)
        opacities: dict for if there are mobjects to give an opacity besides one to. Keys are the mobs, values are the opacities (for set_opacity)
    """
    
    def get_merged_array(mob,attr):
        result = np.array([getattr(mob,attr)])
        for submob in mob.submobjects:
            result = np.append(result, get_merged_array(submob, attr))
        return result
    
    def set_merged_array(mob, attr, array):
            setattr(mob, attr, array[0])
            array = array[1:]
            for submob in mob.submobjects:
                array = set_merged_array(submob, attr, array)
            return array


    # record original opacities, then set them to zero
    hidden_mobs = VGroup(*hidden_mobs)
    strokes = get_merged_array(hidden_mobs,"stroke_opacity")
    fills = get_merged_array(hidden_mobs,"fill_opacity")
    for mob in hidden_mobs:
        mob.set_opacity(0)
        
    # apply updater to any mobs needing it
    for mob in updater_mobs:
        mob.add_updater(updater)
    
    # perform animation!
    animation = animation_func()
    if move_camera_dict == None:
        scene.play(animation,*added_anims,run_time=run_time)
    else:
        scene.move_camera(
            **move_camera_dict, 
            added_anims=[
                animation,
                *added_anims
            ],
            run_time=run_time
        )

    # remove updaters
    for mob in updater_mobs:
        mob.remove_updater(updater)
    
    for mob in hidden_mobs:
        # remove hidden mobs that just got added to the scene because of the animation
        scene.remove(mob)

        # restore opacities
        if mob in opacities:
            mob.set_opacity(opacities[mob])
        else:
            mob.set_opacity(1)
    # set_merged_array(hidden_mobs,"stroke_opacity",strokes)
    # set_merged_array(hidden_mobs,"fill_opacity",fills)






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
        pe = MathTex(r"\mathbf{p}","=",r"p_x \mathbf{x}","=",r"\mathbf{O}\mathbf{v}", font_size=60)        
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

        # flash rejection direction
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

        # merge px and p
        pfe4 = pfe3.copy().shift(DOWN).scale(1.1)
        self.play(
            ReplacementTransform(pfe3[0],pfe4[0]), # p=
            ReplacementTransform(pfe3[1],pfe4[1]), #x
            Merge([pfe3[2][:],coef3[0][3:10]],pfe4[2][:]),
            Merge([pfe3[3][:],coef3[0][10:]],pfe4[3][:]),
            FadeOut(coef3[0][:3]),
            run_time=1.5
        )
        self.wait()

        # projection matrix ellipse
        el = Ellipse(width=5.1,height=1.75).move_to(pfe4[2][3]).rotate(5*DEGREES).shift(RIGHT*0.2+UP*0.15)
        pm = Tex("Projection Matrix", font_size=75).next_to(el,UP)
        self.play(
            DrawBorderThenFill(pm),
            Write(el),
            run_time=1.5
        )
        self.wait(w)

        # projection matrix out
        self.play(
            Unwrite(el),
            DrawBorderThenFill(pm, reverse_rate_function=True),
            run_time=1.5
        )
        self.wait(w)

        # indicate inner then outer product
        self.play(
            Indicate(pfe4[2][1:4],scale_factor=1.75,color=XKCD.LIGHTTURQUOISE)
        ,run_time=1.75)
        self.wait(w)
        self.play(
            Indicate(pfe4[1],scale_factor=2,color=XKCD.BUBBLEGUM),
            Indicate(pfe4[3][:2],scale_factor=2,color=XKCD.BUBBLEGUM)
        ,run_time=1.75)
        self.wait(w)

        # zoom back to diagram
        self.play(frame.animate.move_to(diagram).scale(0.43),run_time=2)
        self.wait()

        # remove r and label
        self.play(FadeOut(r,rl))

        # adjust oblique angle so that x and b overla
        self.remove(rab,db,b,bl,p)
        kv = ValueTracker(k)
        pv = always_redraw(lambda: Arrow(axes.c2p(0,0), axes.c2p(*(xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value())), buff=0, color=PCOLOR))
        bv = always_redraw(lambda: Arrow(axes @ ORIGIN, axes @ (2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])),buff=0, color=BCOLOR))
        blv = always_redraw(lambda: MathTex(r"\mathbf{b}", font_size=60, color=BCOLOR).next_to(bv.get_tip(),buff=0.15))
        dbv = always_redraw(lambda: DashedLine(v.get_end(),axes @ ((2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])) * np.dot(vcoords, (2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])]))) / np.dot((2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])),(2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])))),dash_length=0.1).set_opacity(0.6))
        rabv = always_redraw(lambda: RightAngle(dbv, bv,length=0.2,quadrant=(-1,1)).set_stroke(opacity=0.6))
        self.add(bv,pv,dbv,blv,rabv)
        self.play(
            kv.animate.set_value(1),
            FadeOut(xl),
            run_time=2
        )
        self.wait()
        # other direction
        self.remove(dbv)
        dbv = always_redraw(lambda: DashedLine(v.get_end(),pv.get_end(),dash_length=0.1).set_opacity(0.6))
        self.add(dbv)
        self.play(kv.animate.set_value(1.25),run_time=2)
        self.wait()
        self.play(kv.animate(rate_func=rate_functions.ease_in_sine).set_value(1),run_time=1.5)
        self.remove(dbv)
        dbv = always_redraw(lambda: DashedLine(v.get_end(),axes @ ((2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])) * np.dot(vcoords, (2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])]))) / np.dot((2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])),(2 * np.array([1,-((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[0]) / ((vcoords - (xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords) * kv.get_value()))[1])])))),dash_length=0.1).set_opacity(0.6))
        self.add(dbv)
        self.play(
            kv.animate(rate_func=rate_functions.ease_out_sine).set_value(k),
            FadeIn(xl),
            run_time=1.5
        )
        self.wait()
        self.remove(pv,bv,dbv,blv,rabv)
        self.add(rab,db,b,bl,p)
        self.wait()

        # zoom back out, add stuff back in
        self.play(
            Restore(frame),
            FadeIn(r,rl),
            run_time=2
        )
        self.wait()

        # caption about inner product
        ip1 = MathTex(r"\text{If }",r"\mathbf{b}\cdot \mathbf{x}=1:",font_size=70).next_to(pfe4,UP,aligned_edge=LEFT).shift(UP*0.25)
        color_tex_standard(ip1)
        self.play(Write(ip1[0]),run_time=1.25)
        self.play(Write(ip1[1]),run_time=1.25)
        self.wait()

        # to outer product
        pfe5 = MathTex(r"\mathbf{p} =", r"\mathbf{x}",r"\mathbf{b}^T \mathbf{v}",  font_size=70)
        AlignBaseline(pfe5.move_to(pfe4),pfe4)
        color_tex_standard(pfe5)
        self.play(TransformMatchingTex(pfe4,pfe5),run_time=2)
        self.wait()

        # back to full formula
        self.play(
            TransformMatchingTex(pfe5,pfe4),
            FadeOut(ip1),
            run_time=2)
        self.wait()

        

class Oblique2D(ThreeDScene):
    def construct(self):
        low_plane_resolution = 10 # increase to like 32 or even 64 for higher quality render (but will take way longer)
        high_plane_resolution = 10
        
        x1coords = np.array([1,0,0])
        x2coords = np.array([0,1,0])
        b1coords = np.array([1,0,0.35])
        b2coords = np.array([0,1,0.25])
        vcoords = np.array([0.5,0.7,0.7])
        amatrix = np.vstack([x1coords,x2coords]).T
        bmatrix = np.vstack([b1coords,b2coords]).T
        
        px1coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[0]
        px2coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,amatrix)), np.matmul(bmatrix.T,vcoords))[1]
        pcoords = px1coord*x1coords + px2coord*x2coords
        
        bpx1coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[0]
        bpx2coord = np.matmul(np.linalg.inv(np.matmul(bmatrix.T,bmatrix)), np.matmul(bmatrix.T,vcoords))[1]
        bpcoords = bpx1coord*b1coords + bpx2coord*b2coords

        # define diagram
        axes = ThreeDAxes(
            x_range=[0,0.5],x_length=2.5 / 2,
            y_range=[0,0.5],y_length=2.5 / 2,
            z_range=[0,0.5],z_length=2.5 / 2,
        ).set_opacity(0)        
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR,shade_in_3d=True).set_stroke(width=6)        
        x1 = Arrow(axes @ ORIGIN, axes @ x1coords, buff=0, color=XCOLOR).set_stroke(width=6)
        x2 = Arrow(axes @ ORIGIN, axes @ x2coords, buff=0, color=XCOLOR).set_stroke(width=6)        
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR).set_stroke(width=6)        
        px1 = Arrow(axes @ ORIGIN, axes @ (px1coord*x1coords), buff=0,color=PCOLOR).set_stroke(width=6)
        px2 = Arrow(axes @ ORIGIN, axes @ (px2coord*x2coords), buff=0,color=PCOLOR).set_stroke(width=6)
        dp = DashedLine(v.get_end(),p.get_end(),dash_length=0.15).set_stroke(width=2).set_opacity(0.4)
        dx1 = DashedLine(axes @ pcoords, axes @ (px1coord*x1coords), dash_length=0.15).set_stroke(width=2).set_opacity(0.4)
        dx2 = DashedLine(axes @ pcoords, axes @ (px2coord*x2coords), dash_length=0.15).set_stroke(width=2).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR).set_stroke(width=6)        
        ArrowGradient(r,[PCOLOR,VCOLOR])

        b1 = Arrow(axes @ ORIGIN, axes @ b1coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)
        b2 = Arrow(axes @ ORIGIN, axes @ b2coords, buff=0,color=BCOLOR.lighter()).set_stroke(width=6)

        angle = Arc3d(p.get_center(),r.get_center(),p.get_end(),radius=0.4).set_stroke(opacity=0.4)
        vectors = VGroup(v,x1,x2,p,px1,px2,r).set_shade_in_3d()
        b_vectors = VGroup(b1,b2).set_shade_in_3d()
        dashes = VGroup(dp,dx1,dx2).set_shade_in_3d()

        plane =  Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=low_plane_resolution).set_stroke(width=0.06,opacity=0.5).set_opacity(0.5).set_color(ManimColor('#29ABCA'))
        for mob in [*plane,*x1,*x2,*px1,*px2,*p,*b1,*b2]: mob.z_index_group=Dot()
        
        plane2 = Surface(lambda u,v:axes @ (u*b1coords+v*b2coords),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.06,resolution=low_plane_resolution).set_opacity(0.5).set_color(BCOLOR)

        diagram = Group(axes,plane,vectors,dashes, angle,plane2,b_vectors)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        x1l = MathTex(r"\mathbf{x_1}", color=XCOLOR, font_size=50).next_to(x1.get_end(),DOWN,buff=0.1)
        x2l = MathTex(r"\mathbf{x_2}", color=XCOLOR, font_size=50).next_to(x2.get_end(),DR,buff=0.1)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        px1l = MathTex(r"p_1 \mathbf{x_1}", font_size=40).next_to(px1.get_end(),LEFT,buff=0.15).shift(UP*0.15)
        color_tex_standard(px1l)        
        px2l = MathTex(r"p_2 \mathbf{x_2}", font_size=40).next_to(px2.get_end(),UP,buff=0.15)
        color_tex_standard(px2l)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.5)
        color_tex_standard(rl)
        b1l = MathTex(r"\mathbf{b_1}",font_size=40,color=BCOLOR.lighter()).next_to(b1.get_end(),LEFT)
        b2l = MathTex(r"\mathbf{b_2}",font_size=50,color=BCOLOR.lighter()).next_to(b2.get_end(),RIGHT)
        labels = VGroup(x1l,vl,x2l,pl,px1l,px2l,rl,b1l,b2l)
        diagram.add(labels)                
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.35+RIGHT*0.2)
        self.set_camera_orientation(frame_center=IN*11) # self.set_camera_orientation(zoom=2)
        for vector in vectors+b_vectors+dashes: 
            ArrowStrokeFor3dScene(self,vector,family=True)
        face_camera(self,r)
        ArrowGradient(r,[PCOLOR,VCOLOR])


        # setup plane, vectors
        self.play(Write(plane),run_time=2)
        self.play(GrowArrow(v))
        self.play(Write(vl))
        self.wait()

        # project to p, with dash and angle
        self.play(
            TransformFromCopy(v,p),
            Write(dp),
            run_time=2
        )
        self.play(Write(pl))
        self.play(Write(angle))
        self.wait()

        # a few different projections
        pxc2, pyc2 = 0.3,1.1
        pxc3, pyc3 = 0.2,0.3
        pxc4, pyc4 = 1.1,0.2
        p.save_state()
        ArrowStrokeFor3dScene(self,p.generate_target().put_start_and_end_on(axes @ ORIGIN, axes @ (pxc2*x1coords + pyc2*x2coords)))
        dpv = always_redraw(lambda: ArrowStrokeFor3dScene(self,DashedLine(v.get_end(),p.get_end(),dash_length=0.15).set_stroke(width=2).set_opacity(0.4)))
        self.remove(dp), self.add(dpv)
        anglev = always_redraw(lambda: Arc3d(p.get_center(),dpv.get_center(),p.get_end(),radius=0.4).set_stroke(opacity=0.4))
        self.remove(angle), self.add(anglev)
        self.play(MoveToTarget(p),run_time=1.5)
        ArrowStrokeFor3dScene(self,p.generate_target().put_start_and_end_on(axes @ ORIGIN, axes @ (pxc3*x1coords + pyc3*x2coords)))
        self.play(MoveToTarget(p),run_time=1.5)
        ArrowStrokeFor3dScene(self,p.generate_target().put_start_and_end_on(axes @ ORIGIN, axes @ (pxc4*x1coords + pyc4*x2coords)))
        self.play(MoveToTarget(p),run_time=1.5)
        self.play(Restore(p),run_time=1.5)
        self.remove(dpv,anglev), self.add(dp,angle)
        self.wait()

        # add x,y basis
        self.play(GrowArrow(x1))
        self.play(Write(x1l))
        self.play(GrowArrow(x2))
        self.play(Write(x2l))
        self.wait()

        # to px, py
        self.play(
            TransformFromCopy(p,px1),
            Create(dx1)
        , run_time=1.75)
        self.play(Write(px1l))
        self.play(
            TransformFromCopy(p,px2),
            Create(dx2)
        , run_time=1.75)
        self.play(Write(px2l))
        self.wait(w)

        # zoom out
        alter_diagram(
            self,
            animation_func=lambda:diagram.animate.shift(UP*0.3),
            hidden_mobs=[plane2,b1,b2,b1l,b2l,r,rl],
            opacities={plane2:0.5},
            run_time=1.25,
            move_camera_dict={"frame_center":9.5*IN},
            updater=ArrowStrokeCameraUpdater(self),
            updater_mobs=vectors+b_vectors+dashes,
        )

        # write equation for p
        pe = MathTex(r"\mathbf{p}","=",r"p_1 \mathbf{x_1}","+",r"p_2 \mathbf{x_2}", font_size=60).next_to(diagram,DOWN,buff=0.3)
        color_tex_standard(pe)
        pe[0].set_color(PCOLOR)
        self.play(TransformFromCopy(pl[0],pe[0]),run_time=1.5)
        self.play(Write(pe[1]))
        self.play(
            TransformFromCopy(px1l[0],pe[2]),
            Indicate(px1l[0]),
            run_time=1.5
        )
        self.play(Write(pe[3]))
        self.play(
            TransformFromCopy(px2l[0],pe[4]),
            Indicate(px2l[0]),
            run_time=1.5
        )
        self.wait(w)

        # draw rejection
        self.play(FadeOut(dp))
        self.play(GrowArrow(r),run_time=1.5)
        self.play(Write(rl))
        self.wait()

        # flash rejection
        tr = VMobject().add_points_as_corners(
            [r.get_end(),r.get_start()]
        ).set_color(YELLOW).set_z_index(2)
        mask = Rectangle(height=8,width=12).set_fill(BLACK,0.6).set_stroke(BLACK).move_to(diagram).set_z_index(1)
        self.play(FadeIn(mask),run_time=1.25)
        self.play(ShowPassingFlash(tr,time_width=0.2),run_time=2)
        
        # un-mask and remove stuff
        self.play(
            FadeOut(x1,x1l,x2,x2l,pe,px1l,px2l,dx1,dx2,px1,px2,r,rl),
            FadeOut(mask),
            FadeIn(dp),
            run_time=1.25
        )

        # add plane, zoom in
        self.play(
            Write(plane2)
        )



        alter_diagram(
            self,
            lambda: diagram.animate.rotate(40*DEGREES, axis=(axes @ (vcoords - pcoords))-(axes @ ORIGIN), about_point=diagram.get_center()).rotate(10*DEGREES,axis=(axes @ b2coords) - (axes @ ORIGIN),about_point=diagram.get_center()).shift(DOWN*0.15),
            hidden_mobs=[x1,x2,x1l,x2l,pe,px1l,px2l,dx1,dx2,px1,px2,r,rl,b1,b2,b1l,b2l],
            opacities={dx1:0.4,dx2:0.4},
            run_time=2,
            move_camera_dict={"frame_center":IN*13},
            updater=ArrowStrokeCameraUpdater(self),
            updater_mobs=vectors+b_vectors+dashes,
        )
                

        # add right angle
        ra = VGroup(
            Line(axes @ (0.9*bpcoords),axes @ (0.9*bpcoords+0.1*(vcoords-pcoords)), stroke_width=2),
            Line(axes @ (0.9*bpcoords+0.1*(vcoords-pcoords)),axes @ (1*bpcoords+0.1*(vcoords-pcoords)), stroke_width=2)
        ).set_stroke(opacity=0.6)
        self.play(ReplacementTransform(ra.copy().scale(20).shift(LEFT*3).set_opacity(0),ra),run_time=2)
        self.wait()

        # move the oblique plane around
        # these value trackers will store the temporary basis vectors
        b1v = [ValueTracker(b1coords[0]),ValueTracker(b1coords[1]),ValueTracker(b1coords[2])]
        b2v = [ValueTracker(b2coords[0]),ValueTracker(b2coords[1]),ValueTracker(b2coords[2])]
        # these functions calculate where things need to be dynamically
        def get_b_values(b1,b2):
            return (
                np.array([b1[0].get_value(),b1[1].get_value(),b1[2].get_value()]),
                np.array([b2[0].get_value(),b2[1].get_value(),b2[2].get_value()])
            )
        def bm_calc(b1,b2):    
            bmatrix = np.vstack([b1,b2]).T
            return bmatrix
        def px_calc(b1,b2):
            px = np.matmul(np.linalg.inv(np.matmul(bm_calc(b1,b2).T,amatrix)), np.matmul(bm_calc(b1,b2).T,vcoords))[0]
            return px
        def py_calc(b1,b2):
            py = np.matmul(np.linalg.inv(np.matmul(bm_calc(b1,b2).T,amatrix)), np.matmul(bm_calc(b1,b2).T,vcoords))[1]
            return py
        def p_calc(b1,b2):
            p = px_calc(b1,b2)*x1coords + py_calc(b1,b2)*x2coords
            return p
        def bx_calc(b1,b2):
            bx = np.matmul(np.linalg.inv(np.matmul(bm_calc(b1,b2).T,bm_calc(b1,b2))), np.matmul(bm_calc(b1,b2).T,vcoords))[0]
            return bx
        def by_calc(b1,b2):
            by = np.matmul(np.linalg.inv(np.matmul(bm_calc(b1,b2).T,bm_calc(b1,b2))), np.matmul(bm_calc(b1,b2).T,vcoords))[1]
            return by
        def bp_calc(b1,b2):
            bp = b1*bx_calc(b1,b2) + b2 * by_calc(b1,b2)
            return bp
        # remove things and replace them with copies that will update
        self.remove(dp,ra,angle,plane,plane2)
        p.add_updater(lambda mob: ArrowStrokeFor3dScene(self,mob.put_start_and_end_on(axes @ ORIGIN, axes @ (px_calc(*get_b_values(b1v,b2v))*x1coords + py_calc(*get_b_values(b1v,b2v))*x2coords))))
        pl.add_updater(lambda mob: mob.next_to(p.get_end(),DR,buff=0.15))
        dpv = always_redraw(lambda: ArrowStrokeFor3dScene(self,DashedLine(v.get_end(),p.get_end(),dash_length=0.15,shade_in_3d=True).set_stroke(width=2).set_opacity(0.4)))
        anglev = always_redraw(lambda: Arc3d(p.get_center(),dpv.get_start(),p.get_end(),radius=0.4).set_stroke(opacity=0.4).set_shade_in_3d(True))
        plane2v = always_redraw(lambda: Surface(lambda u,v:axes @ (u*(get_b_values(b1v,b2v)[0])+v*(get_b_values(b1v,b2v)[1])),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.06,resolution=high_plane_resolution).set_opacity(0.5).set_color(BCOLOR))
        plane1v = always_redraw(lambda: Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],stroke_width=0.06,resolution=high_plane_resolution).set_opacity(0.5).set_color(ManimColor('#29ABCA')).set_z_index_group(Dot()))
        rav = always_redraw(lambda: VGroup(
            Line(axes @ (0.9*bp_calc(*get_b_values(b1v,b2v))),axes @ (0.9*bp_calc(*get_b_values(b1v,b2v))+0.1*(vcoords-p_calc(*get_b_values(b1v,b2v)))), stroke_width=2),
            Line(axes @ (0.9*bp_calc(*get_b_values(b1v,b2v))+0.1*(vcoords-p_calc(*get_b_values(b1v,b2v)))),axes @ (1*bp_calc(*get_b_values(b1v,b2v))+0.1*(vcoords-p_calc(*get_b_values(b1v,b2v)))), stroke_width=2)
        ).set_stroke(opacity=0.6).set_shade_in_3d(True))
        self.add(plane1v,anglev,dpv,plane2v,rav)
        # these are the sets of coordinates for the basis vectors for the transformations
        b1c2 = np.array([1,0,0.4])
        b2c2 = np.array([0,1,0])
        b1c3 = np.array([1,0,0])
        b2c3 = np.array([0,1,0.4])
        self.play(
            b1v[0].animate.set_value(b1c2[0]),b1v[1].animate.set_value(b1c2[1]),b1v[2].animate.set_value(b1c2[2]),
            b2v[0].animate.set_value(b2c2[0]),b2v[1].animate.set_value(b2c2[1]),b2v[2].animate.set_value(b2c2[2]),
            run_time=2
        )
        self.play(
            b1v[0].animate.set_value(b1c3[0]),b1v[1].animate.set_value(b1c3[1]),b1v[2].animate.set_value(b1c3[2]),
            b2v[0].animate.set_value(b2c3[0]),b2v[1].animate.set_value(b2c3[1]),b2v[2].animate.set_value(b2c3[2]),
            run_time=2
        )
        # go back home
        self.play(
            b1v[0].animate.set_value(b1coords[0]),b1v[1].animate.set_value(b1coords[1]),b1v[2].animate.set_value(b1coords[2]),
            b2v[0].animate.set_value(b2coords[0]),b2v[1].animate.set_value(b2coords[1]),b2v[2].animate.set_value(b2coords[2]),
            run_time=2
        )
        self.remove(plane1v,anglev,dpv,plane2v,rav)
        self.add(dp,ra,angle,plane,plane2)
        p.clear_updaters(), pl.clear_updaters()
        self.wait()

        # pan to plane 2, fade the rest out
        o = Dot().shift(OUT*0.2)
        self.move_camera(
            frame_center=self.camera.frame_center+UP*0.75,
            added_anims=
            [mob.animate.set_z_index_group(o) for mob in plane2]
            +[FadeOut(plane,p,pl,angle,dp[-4:])],
            run_time=2
        )
        
        # draw basis vectors for plane 2
        self.play(GrowArrow(b1))
        self.play(Write(b1l))
        self.play(GrowArrow(b2))
        self.play(Write(b2l))
        self.wait()

        # back to both planes
        off = VGroup(x1,x1l,x2,x2l,px1,px1l,px2,px2l,r,rl).set_opacity(0)
        VGroup(dx1,dx2).set_opacity(0)
        for mob in vectors+b_vectors+dashes: mob.add_updater(ArrowStrokeCameraUpdater(self))
        self.move_camera(
            frame_center=self.camera.frame_center+DOWN*0.75+OUT*3.5,
            added_anims=
            [mob.animate.set_z_index_group(mob) for mob in plane2]
            +[FadeIn(plane,p,pl,angle,dp[-3:],pe),
              diagram.animate.shift(UP*0.15).rotate(-10*DEGREES,axis=(axes @ b2coords) - (axes @ ORIGIN),about_point=diagram.get_center()).rotate(-40*DEGREES, axis=(axes @ (vcoords - pcoords))-(axes @ ORIGIN), about_point=diagram.get_center()),
              FadeOut(ra)],
            run_time=2
        )
        for mob in vectors+b_vectors+dashes: mob.clear_updaters()
        self.play(
            off.animate.set_opacity(1),
            VGroup(dx1,dx2).animate.set_opacity(0.4),
            FadeOut(dp),
            run_time=2)
        self.wait()
        
        # zoom out
        alter_diagram(
            self,
            lambda: diagram.animate.to_corner(UR),
            hidden_mobs=[dp],
            run_time=2,
            move_camera_dict={"frame_center":ORIGIN},
            updater=ArrowStrokeCameraUpdater(self),
            updater_mobs=list(vectors) + list(b_vectors) + list(dashes),
            added_anims=[pe.animate.next_to(diagram.copy().to_corner(UR),DOWN)],
            opacities={dp:0.4}
        )
        self.wait()

        # the code below is copied from formulas, just lightly updated
        # to normal equations
        nex = AlignBaseline(MathTex(r"(\mathbf{v}-\mathbf{p})\cdot \mathbf{b_1} =0", font_size=65).shift(2*LEFT),pe)
        color_tex_standard(nex)
        ney = MathTex(r"(\mathbf{v}-\mathbf{p})\cdot \mathbf{b_2} =0", font_size=65).next_to(nex,DOWN).shift(DOWN)
        color_tex_standard(ney)
        self.play(
            Write(nex[0][0]), Write(nex[0][4]), Write(ney[0][0]), Write(ney[0][4]), # parentheses
            ReplacementTransform(rl[0].copy(),nex[0][1:4]), ReplacementTransform(rl[0].copy(),ney[0][1:4]) # v-p
        , run_time=2)
        self.play(Write(nex[0][5]), Write(ney[0][5])) # dot
        self.play(ReplacementTransform(b1l[0].copy(),nex[0][6:8]), ReplacementTransform(b2l[0].copy(),ney[0][6:8]), run_time=2) # b1,b2
        self.play(Write(nex[0][-2:]), Write(ney[0][-2:]),run_time=1.5) # =0        
        self.wait(w)

        # rearrange normal equations
        nex1 = MathTex(r"\mathbf{b_1} \cdot \mathbf{p}","=",r"\mathbf{b_1} \cdot \mathbf{v}", font_size=65).move_to(nex)
        AlignBaseline(nex1,nex)
        color_tex_standard(nex1)
        ney1 = MathTex(r"\mathbf{b_2} \cdot \mathbf{p}","=",r"\mathbf{b_2} \cdot \mathbf{v}", font_size=65).move_to(ney)
        AlignBaseline(ney1,ney)
        color_tex_standard(ney1)
        self.play(*TransformBuilder(
            nex,nex1,
            [
                ([0,[0,4]],None), # ()
                ([0,1],[-1,-1],None,{"path_arc":120*DEGREES}), # v
                ([0,2],None), # -
                ([0,3],[0,3]), # p
                ([0,5],[0,2],None,{"path_arc":120*DEGREES}), # dot
                ([0,5],[2,2], TransformFromCopy,{"path_arc":-120*DEGREES}), #dot
                ([0,[6,7]],[0,[0,1]],None,{"path_arc":120*DEGREES}), # b1
                ([0,[6,7]],[2,[0,1]],TransformFromCopy), # b1
                ([0,8],[1,0]), # =
                ([0,-1],None) # 0
            ]
        ), run_time=2.5)
        self.play(*TransformBuilder(
            ney,ney1,
            [
                ([0,[0,4]],None), # ()
                ([0,1],[-1,-1],None,{"path_arc":120*DEGREES}), # v
                ([0,2],None), # -
                ([0,3],[0,3]), # p
                ([0,5],[0,2],None,{"path_arc":120*DEGREES}), # dot
                ([0,5],[2,2], TransformFromCopy,{"path_arc":-120*DEGREES}), #dot
                ([0,[6,7]],[0,[0,1]],None,{"path_arc":120*DEGREES}), # b2
                ([0,[6,7]],[2,[0,1]],TransformFromCopy), # b2
                ([0,8],[1,0]), # =
                ([0,-1],None) # 0
            ]
        ), run_time=2.5)
        self.wait(w)

        # substitute p
        nex2 = MathTex(r"\mathbf{b_1} \cdot (p_1 \mathbf{x_1} + p_2 \mathbf{x_2})","=",r"\mathbf{b_1} \cdot \mathbf{v}", font_size=65).move_to(nex1)        
        color_tex_standard(nex2)
        AlignBaseline(nex2,nex1)
        ney2 = MathTex(r"\mathbf{b_2} \cdot (p_1 \mathbf{x_1} + p_2 \mathbf{x_2})","=",r"\mathbf{b_2} \cdot \mathbf{v}", font_size=65).move_to(ney1)        
        color_tex_standard(ney2)
        AlignBaseline(ney2,ney1)
        self.play(
            ReplacementTransform(nex1[0][0:2], nex2[0][0:2]), # b1
            ReplacementTransform(nex1[0][2], nex2[0][2]), # dot
            FadeIn(nex2[0][3]), FadeIn(nex2[0][-1]), # ()
            FadeOut(nex1[0][3]), # p
            TransformFromCopy(pe[2][:],nex2[0][4:8]), # px x
            TransformFromCopy(pe[3][0], nex2[0][8]), # +
            TransformFromCopy(pe[4][:],nex2[0][9:13]), # py y
            ReplacementTransform(nex1[1], nex2[1]), # =
            ReplacementTransform(nex1[2], nex2[2]), # rhs
            ReplacementTransform(ney1[0][0:2], ney2[0][0:2]), # b2
            ReplacementTransform(ney1[0][2], ney2[0][2]), # dot
            FadeIn(ney2[0][3]), FadeIn(ney2[0][-1]), # ()
            FadeOut(ney1[0][3]), # p
            ReplacementTransform(pe[2][:],ney2[0][4:8]), # px x
            ReplacementTransform(pe[3][0], ney2[0][8]), # +
            ReplacementTransform(pe[4][:],ney2[0][9:13]), # py y
            ReplacementTransform(ney1[1], ney2[1]), # =
            ReplacementTransform(ney1[2], ney2[2]), # rhs            
            FadeOut(pe[:2]),
            run_time=3
        )
        self.remove(pe)
        self.wait()   

        # distribute dot products
        nex3 = MathTex(r"p_1 \mathbf{b_1} \cdot \mathbf{x_1} + p_2 \mathbf{b_1} \cdot \mathbf{x_2}","=",r"\mathbf{b_1} \cdot \mathbf{v}", font_size=65).move_to(nex1).shift(RIGHT*0.2)       
        color_tex_standard(nex3)
        AlignBaseline(nex3,nex1)
        ney3 = MathTex(r"p_1 \mathbf{b_2} \cdot \mathbf{x_1} + p_2 \mathbf{b_2} \cdot \mathbf{x_2}","=",r"\mathbf{b_2} \cdot \mathbf{v}", font_size=65).move_to(ney1).shift(RIGHT*0.2)    
        color_tex_standard(ney3)
        AlignBaseline(ney3,ney1)
        self.play(*TransformBuilder(
            nex2,nex3,
            [
                ([0,[0,1]],[0,[2,3]],None,{"path_arc":-280*DEGREES}), # b1
                ([0,[0,1]],[0,[10,11]],TransformFromCopy,{"path_arc":120*DEGREES}), # b1
                ([0,2],[0,4],None,{"path_arc":-280*DEGREES}), # dot
                ([0,2],[0,12],TransformFromCopy,{"path_arc":120*DEGREES}), # dot
                ([0,3],None), ([0,-1],None), # ()
                ([0,[4,5]], [0,[0,1]]), # px
                ([0,[6,7]],[0,[5,6]]), # x
                ([0,8],[0,7]), # +
                ([0,[9,10]],[0,[8,9]]), #py
                ([0,[11,12]],[0,[13,14]]), # y
                (1,1), (2,2) # = rhs
            ]
        )
        ,run_time=2)
        self.play(*TransformBuilder(
            ney2,ney3,
            [
                ([0,[0,1]],[0,[2,3]],None,{"path_arc":-280*DEGREES}), # b2
                ([0,[0,1]],[0,[10,11]],TransformFromCopy,{"path_arc":120*DEGREES}), # b2
                ([0,2],[0,4],None,{"path_arc":-280*DEGREES}), # dot
                ([0,2],[0,12],TransformFromCopy,{"path_arc":120*DEGREES}), # dot
                ([0,3],None), ([0,-1],None), # ()
                ([0,[4,5]], [0,[0,1]]), # px
                ([0,[6,7]],[0,[5,6]]), # x
                ([0,8],[0,7]), # +
                ([0,[9,10]],[0,[8,9]]), #py
                ([0,[11,12]],[0,[13,14]]), # y
                (1,1), (2,2) # = rhs
            ]
        )
        ,run_time=2)
        self.wait(w)


        # to matrix equations
        gram = Matrix([
            [r"\mathbf{b_1} \cdot \mathbf{x_1}",r"\mathbf{b_1} \cdot \mathbf{x_2}"],
            [r"\mathbf{b_2} \cdot \mathbf{x_1}",r"\mathbf{b_2} \cdot \mathbf{x_2}"],
        ],element_alignment_corner=UL, h_buff=2)
        color_tex_standard(gram)
        comps = Matrix([
            ["p_1"],
            ["p_2"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65}).next_to(gram)
        color_tex_standard(comps)
        eq = MathTex("=").next_to(comps)
        dots = Matrix([
            [r"\mathbf{b_1} \cdot \mathbf{v}"],
            [r"\mathbf{b_2} \cdot \mathbf{v}"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65}).next_to(eq)
        color_tex_standard(dots)
        VGroup(gram,comps,eq,dots).arrange().shift(DOWN*1)
        self.play(
            Merge([nex3[0][0:2],ney3[0][0:2]], comps.get_entries()[0][0][:]), # px
            Merge([nex3[0][8:10],ney3[0][8:10]], comps.get_entries()[1][0][:]), # py
            ReplacementTransform(nex3[0][2:7],gram.get_rows()[0][0][0][:]), # b1.x
            ReplacementTransform(nex3[0][10:15],gram.get_rows()[0][1][0][:]), # b1.y
            ReplacementTransform(ney3[0][2:7],gram.get_rows()[1][0][0][:]), # b2.x
            ReplacementTransform(ney3[0][10:15],gram.get_rows()[1][1][0][:]), # b2.y
            ReplacementTransform(nex3[2],dots.get_entries()[0][0]), # b1.v
            ReplacementTransform(ney3[2],dots.get_entries()[1][0]), # b2.v            
            FadeIn(gram.get_brackets(),comps.get_brackets(),dots.get_brackets()), # brackets
            Merge([nex3[1],ney3[1]],eq[0]), # =
            FadeOut(nex3[0][7],shift=DOWN),
            FadeOut(ney3[0][7]),
            run_time=3
        )
        self.wait()

        # from dots to transpose
        gramt = Matrix([
            [r"\mathbf{b_1}^T \mathbf{x_1}",r"\mathbf{b_1}^T \mathbf{x_2}"],
            [r"\mathbf{b_2}^T \mathbf{x_1}",r"\mathbf{b_2}^T \mathbf{x_2}"],
        ],element_alignment_corner=UL,h_buff=2).move_to(gram)
        color_tex_standard(gramt)
        dotst = Matrix([
            [r"\mathbf{b_1}^T \mathbf{v}"],
            [r"\mathbf{b_2}^T \mathbf{v}"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65}).move_to(dots)
        color_tex_standard(dotst)
        self.play(
            ReplacementTransform(gram,gramt),
            ReplacementTransform(dots,dotst)
        ,run_time=1.75)
        self.wait(w)

        # factor
        xtyt = Matrix([
            [r"\mathbf{b_1}^T"],
            [r"\mathbf{b_2}^T"]
        ])
        color_tex_standard(xtyt)
        xy = Matrix([[r"\mathbf{x_1}",r"\mathbf{x_2}"]], element_alignment_corner=UL)
        color_tex_standard(xy)        
        xtyt4v = Matrix([
            [r"\mathbf{b_1}^T"],
            [r"\mathbf{b_2}^T"]
        ])
        color_tex_standard(xtyt4v)
        veq = MathTex(r"\mathbf{v}",font_size=75)
        color_tex_standard(veq)        
        VGroup(xtyt,xy,comps.generate_target(),eq.generate_target(),xtyt4v,veq).arrange().shift(DOWN)
        VGroup(xtyt,xy).next_to(comps,LEFT)
        self.play(
            ReplacementTransform(gramt.get_brackets()[0], xtyt.get_brackets()[0]), # left [
            FadeIn(xtyt.get_brackets()[1]), # left ]
            FadeIn(xy.get_brackets()[0]), # right [
            ReplacementTransform(gramt.get_brackets()[1], xy.get_brackets()[1]), # right ]
            *[Merge([entry[0][i] for entry in gramt.get_rows()[0]], xtyt.get_entries()[0][0][i]) for i in [0,1,2]], # xt
            *[Merge([entry[0][i] for entry in gramt.get_rows()[1]], xtyt.get_entries()[1][0][i]) for i in [0,1,2]], # yt
            Merge([entry[0][3:5] for entry in gramt.get_columns()[0]], xy.get_entries()[0]), # x
            Merge([entry[0][3:5] for entry in gramt.get_columns()[1]], xy.get_entries()[1]) # y            
        ,run_time=3.5)
        self.play(            
            VGroup(xtyt,xy).animate.next_to(comps.target,LEFT),
            MoveToTarget(comps),
            MoveToTarget(eq),
            ReplacementTransform(dotst.get_brackets(),xtyt4v.get_brackets()), 
            *[ReplacementTransform(entry1[0][:3], entry2[0][:3]) for entry1,entry2 in zip(dotst.get_entries(), xtyt4v.get_entries())],
            Merge([entry[0][3] for entry in dotst.get_entries()], veq[0][0]) # v
        ,run_time=2.5)
        self.wait(w)

        # write A at bottom, and replace xy matrix with A
        aeq = MathTex("A","=")
        xyadef = xy.copy().next_to(aeq)
        xyadef.shift(UP*(GetBaseline(aeq) - GetBaseline(xyadef.get_entries()[0])))
        VGroup(aeq,xyadef).next_to(VGroup(xtyt,xy,comps,eq,xtyt4v,veq),DOWN*2)
        self.play(Write(aeq[0]))
        self.play(Write(aeq[1]))
        A = MathTex("A", font_size=75).move_to(xy)
        self.play(
            ReplacementTransform(xy,xyadef),
            TransformFromCopy(aeq[0],A[0],path_arc=180*DEGREES)
        , run_time=2)

        # to b matrix
        beq = MathTex("B","=")
        bdef = Matrix([[r"\mathbf{b_1}",r"\mathbf{b_2}"]], element_alignment_corner=UL)
        color_tex_standard(bdef)    
        VGroup(VGroup(beq,bdef).arrange(),VGroup(aeq.generate_target(),xyadef.generate_target())).arrange(buff=1).move_to(VGroup(aeq,xyadef))    
        AlignBaseline(beq,aeq)
        bdef.shift(UP*(GetBaseline(beq) - GetBaseline(bdef.get_entries()[0])))
        # xyadef.target.shift(UP*(GetBaseline(aeq.target) - GetBaseline(xyadef.target.get_entries()[0])))
        Bt =  MathTex("B^T", font_size=75).move_to(xtyt)
        self.play(
            MoveToTarget(aeq),
            MoveToTarget(xyadef),
            run_time=1.5
        )
        self.play(Write(beq[0]))
        self.play(Write(beq[1]))        
        self.play(
            TransformFromCopy(beq[0][0],Bt[0][0],path_arc=120*DEGREES), # B
            Merge([entry[0][2] for entry in xtyt.get_entries()], Bt[0][1]), # ^T
            ReplacementTransform(xtyt.get_brackets(),bdef.get_brackets()), # brackets
            *[ReplacementTransform(entry1[0][:2], entry2[0][:2]) for entry1, entry2 in zip(xtyt.get_entries(), bdef.get_entries())], #b1, b2
            run_time=2.25
        )
        Bt4v = MathTex("B^T", font_size=75).move_to(xtyt4v)
        AlignBaseline(Bt4v,A)
        self.play(ReplacementTransform(xtyt4v,Bt4v), run_time=2)
        
        # collect equation
        VGroup(Bt.generate_target(),A.generate_target(),comps.generate_target(),eq.generate_target(),Bt4v.generate_target(),veq.generate_target()).arrange(buff=0.15).shift(DOWN)
        for mob in [Bt,eq,Bt4v,veq]: AlignBaseline(mob.target,A.target)
        self.play(
            *[MoveToTarget(mob) for mob in [Bt,A,comps,eq,Bt4v,veq]]
        ,run_time=2)
        self.wait(w)

        # invert Bta
        compform = MathTex("=",r"\left(B^T A \right)^{-1}","B^T",r"\mathbf{v}", font_size=75)
        color_tex_standard(compform)
        VGroup(comps.generate_target(),compform).arrange().shift(DOWN)
        self.play(
            MoveToTarget(comps),
            ReplacementTransform(eq[0],compform[0]), #=            
            FadeIn(compform[1][0]), FadeIn(compform[1][-3:]), # ()-1
            ReplacementTransform(Bt[0][:],compform[1][1:3],path_arc=-230*DEGREES), # Bt
            ReplacementTransform(A[0][0],compform[1][3],path_arc=-230*DEGREES), # A
            ReplacementTransform(Bt4v[0],compform[2]), # Bt
            ReplacementTransform(veq[0],compform[-1]) # v            
        , run_time=2.5)
        self.wait(w)

        # move components
        compsfull = VGroup(comps,compform)
        self.play(compsfull.animate.move_to(LEFT*3+UP*2),run_time=1.75)

        # write p formula
        pe = MathTex(r"\mathbf{p}","=",r"p_1 \mathbf{x_1}","+",r"p_2 \mathbf{x_2}", font_size=80)
        color_tex_standard(pe).shift(DOWN*0.5)
        self.play(FadeIn(pe,shift=RIGHT),run_time=1.25)
        self.wait(w)

        # factor p
        peq = MathTex(r"\mathbf{p}","=", font_size=80)
        color_tex_standard(peq)
        xy = Matrix([[r"\mathbf{x_1}",r"\mathbf{x_2}"]], element_alignment_corner=UL)
        color_tex_standard(xy)
        compsp = Matrix([
            ["p_1"],
            ["p_2"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65})
        color_tex_standard(compsp)
        VGroup(peq,xy,compsp).arrange().shift(DOWN*0.5)
        self.play(
            ReplacementTransform(pe[0],peq[0]), # p
            ReplacementTransform(pe[1],peq[1]), # =
            FadeIn(xy.get_brackets()[0],compsp.get_brackets()[1]),
            ReplacementTransform(pe[2][0:2],compsp.get_entries()[0][0][:]), # px
            ReplacementTransform(pe[4][0:2],compsp.get_entries()[1][0][:]), # py
            ReplacementTransform(pe[2][2:4], xy.get_entries()[0][0][0:2]), # x1
            ReplacementTransform(pe[4][2:4], xy.get_entries()[1][0][0:2]), # x2
            ReplacementTransform(pe[3],VGroup(xy.get_brackets()[1],compsp.get_brackets()[0])) # +
        ,run_time=3)
        self.wait(w)

        # first term to A
        A = MathTex("A", font_size=80).move_to(xy)
        AlignBaseline(A,peq)
        self.play(
            FadeOut(xy),
            TransformFromCopy(aeq[0],A)
        , run_time=1.75)
        self.wait(w)

        # substitute in components
        pe2 = MathTex(r"\mathbf{p}","=","A",r"\left(B^T A \right)^{-1}","B^T",r"\mathbf{v}",font_size=80)
        color_tex_standard(pe2).shift(DOWN*0.5)
        self.play(
            ReplacementTransform(peq[0],pe2[0]), # p
            ReplacementTransform(peq[1],pe2[1]), # =
            ReplacementTransform(A[0],pe2[2]), # A
            FadeOut(compsp), # vector components
            TransformFromCopy(compform[1:], pe2[3:]) # ata-1atv
        ,run_time=2.5)
        self.wait(w)



class BackTo1d(MovingCameraScene):
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
        vectors = VGroup(x,v,p)     

        dx = DashedLine(v.get_end(),p.get_end(),dash_length=0.1).set_opacity(0.6)
        angle = Angle(p,dx,radius=0.35,quadrant=(-1,-1),other_angle=True) 


        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DR,buff=0.03)        
        labels = VGroup(xl, vl, pl)
        
        diagram = VGroup(axes, vectors, labels,angle,dx)
        
        frame.scale(0.5)

        self.add(diagram)
        
        # blur in, in post

        self.wait()

        # write projector equation
        self.play(diagram.animate.shift(LEFT*0.75))
        ppv = MathTex(r"\mathbf{p}","=",r"\mathbf{O}",r"\mathbf{v}",font_size=60).shift(UP*1+RIGHT*1.5)
        color_tex_standard(ppv)
        self.play(TransformFromCopy(pl[0],ppv[0]),run_time=1.5)
        self.play(Write(ppv[1]),run_time=1.25)
        self.play(
            FadeIn(ppv[2],shift=LEFT),
            TransformFromCopy(vl[0],ppv[3]),
            run_time=2
        )
        self.wait()

        # projector equation splits into 2 underlines
        ppv2 = AlignBaseline(MathTex(r"\mathbf{p}","=",r"\underline{\hspace{0.2cm}} \, \underline{\hspace{0.2cm}}",r"\mathbf{v}",font_size=60).move_to(ppv),ppv)
        color_tex_standard(ppv2)
        self.play(ReplacementTransform(ppv,ppv2),run_time=1.5)
        self.wait()

        from PIL import Image
        img =  ImageMobject(Image.fromarray(self.renderer.camera.pixel_array)).scale_to_fit_width(config.frame_width).scale(0.5)
        self.add(img)
        self.play(ImageBlur(img))
        self.wait()

        shear = Tex("Shear",font_size=80)
        self.play(DrawBorderThenFill(shear))
        self.wait()

        # self.play(FadeOut(*self.mobjects))
        

    
class ShearIntro(LinearTransformationScene):
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=False,
            leave_ghost_vectors=False,
            **kwargs
        )

    def construct(self):
        self.wait()
        
        # indicate horizontal axis
        tr = VMobject().add_points_as_corners(
            [[7,0,0],[-7,0,0]]
        ).set_color(YELLOW)
        self.play(ShowPassingFlash(tr),run_time=2)
        self.wait()

        # first shear example
        matrix = [[1, -1], [0, 1]]
        self.apply_matrix(matrix)
        self.wait()

        # undo shear
        self.apply_inverse(matrix)

        # vertical passing flash
        tr = VMobject().add_points_as_corners(
            [[0,-4,0],[0,4,0]]
        ).set_color(YELLOW)
        self.play(ShowPassingFlash(tr),run_time=2)
        self.remove(tr)

        # vertical shear
        print(self.mobjects)
        matrix = [[1,0],[1,1]]
        self.apply_matrix(matrix)
        self.wait()

        # diagonal shear

        


# config.from_animation_number = 80
# config.upto_animation_number = 71


"""
with tempconfig({
        "quality": "medium_quality",
        "from_animation_number":config.from_animation_number,
        "upto_animation_number":config.upto_animation_number
    }):
    scene = Oblique2D()
    scene.render()
"""


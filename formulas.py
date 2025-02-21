from manim import *
from scipy import stats
import numpy as np

from utils import *
# from manim import Arrow3D, Cone # over-writing the over-writing from utils :/

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




class DotProduct(Scene):
    def construct(self):
        # title
        dp = Tex("Dot Product", font_size=75)
        ul = Line(dp.get_corner(DL)+DL*0.2, dp.get_corner(DR)+DR*0.2, color=UNDERLINE_COLOR)        
        self.play(Write(dp))
        self.play(Write(ul))
        self.wait()
        self.play(VGroup(dp,ul).animate.to_edge(UP))
        self.wait()

        # a dot b 
        abdct = MathTex(r"\mathbf{a}",r"\cdot",r"\mathbf{b}","=",r"|\mathbf{a}|",r"|\mathbf{b}|",r"cos\theta", font_size=65)
        ab = AlignBaseline(MathTex(r"\mathbf{a}",r"\cdot",r"\mathbf{b}", font_size=65),abdct)
        self.play(Write(ab))
        self.wait()

        # = ab cos theta
        self.play(ReplacementTransform(ab[:], abdct[:3]))
        self.play(Write(abdct[3]))
        for part in abdct[4:]:
            self.play(Write(part))        
        self.wait(w)

        # to column vectors
        a = Matrix([["a_1"],["a_2"],["a_3"]])
        dot = MathTex(r"\cdot", font_size=65)
        b = Matrix([["b_1"],["b_2"],["b_3"]])
        VGroup(a,dot,b).arrange()        
        ab = AlignBaseline(MathTex(r"\mathbf{a}",r"\cdot",r"\mathbf{b}","=", font_size=65).next_to(a,LEFT),abdct)
        ct = AlignBaseline(MathTex("=",r"|\mathbf{a}|",r"|\mathbf{b}|",r"cos\theta", font_size=65).next_to(b),abdct)
        self.play(
            ReplacementTransform(abdct[:4], ab[:4]),
            TransformFromCopy(abdct[3], ct[0]),
            ReplacementTransform(abdct[4:], ct[1:])
        )
        self.play(TransformFromCopy(ab[0],a))
        self.play(TransformFromCopy(ab[1],dot))
        self.play(TransformFromCopy(ab[2],b))        
        self.wait(w)

        # to multiply components
        comps = AlignBaseline(MathTex("=","a_1 b_1","+","a_2 b_2","+","a_3 b_3", font_size=65).next_to(b), abdct)
        for mob in [a, dot, b, ct]:
            mob.target = mob.copy()
        VGroup(a.target,dot.target,b.target,comps,ct.target).arrange()
        self.play(*[
            MoveToTarget(mob)
                for mob in [a,dot,b,ct]
            ],
            FadeOut(ab[:3]),
            ReplacementTransform(ab[3],comps[0])
            )
        for i in [0,1,2]:
            self.play(
                Indicate(a[0][i]), Indicate(b[0][i]),
                TransformFromCopy(a[0][i],comps[2*i+1][:2]), 
                TransformFromCopy(b[0][i],comps[2*i+1][2:]),
            )
            if i<2: self.play(Write(comps[2*i+2]))
        self.wait(w)

        self.play(FadeOut(*self.mobjects))



class Project1d(MovingCameraScene):
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
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r).shift(UP*0.2+LEFT*0.3)
        color_tex_standard(rl)
        diagram = VGroup(axes,x,v,xl,vl,r,rl).to_corner(UR)
        frame.move_to(diagram)

        # start drawing diagram
        self.play(GrowArrow(x))
        self.play(GrowArrow(v))       
        self.play(Write(xl))
        self.play(Write(vl))
        self.wait(w)

        # zoom in        
        self.play(frame.animate.scale(0.5).move_to(VGroup(x,v).get_center()), run_time=2)

        # draw p and label
        p = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)
        self.play(TransformFromCopy(v,p), run_time=2)
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DOWN)        
        self.play(Write(pl))
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
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0)
        ArrowGradient(r,[PCOLOR,VCOLOR])
        self.play(GrowArrow(r))
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r).shift(UP*0.2+LEFT*0.3)
        color_tex_standard(rl)
        self.play(Write(rl))
        self.wait(w)

        # rejection text
        rej = Tex("Rejection").next_to(rl,UP).shift(DOWN*0.15)#.align_to(rl,LEFT)
        self.play(DrawBorderThenFill(rej))
        self.wait(w)
        self.play(DrawBorderThenFill(rej, reverse_rate_function=True))
        self.wait(w)

        # perp
        ra = RightAngle(p,r,length=0.2,quadrant=(-1,1))
        self.play(Write(ra))
        self.wait(w)

        # dashed line
        dl = DashedLine(axes.c2p(*(vcoords-pcoords)*-1.5),axes.c2p(*(vcoords-pcoords)*2.5),dash_length=0.2,z_index=-1)
        ra2 = RightAngle(p,dl,length=0.2,quadrant=(1,1))
        ra.save_state()
        self.play(
            Create(dl),
            Transform(ra,ra2)
        , run_time=1.5)        
        
        # project rejection
        r2 = Arrow(axes.c2p(0,0), axes.c2p(*(vcoords-pcoords)), buff=0)
        ArrowGradient(r2,[PCOLOR,VCOLOR])
        self.play(TransformFromCopy(v,r2), run_time=2)
        r.save_state()
        self.play(r.animate.move_to(r2), run_time=1.5)
        self.remove(r2)
        self.wait(w)

        # restore
        self.play(
            Restore(r),
            Restore(ra),
            FadeOut(dl)
        ,run_time=2)
        self.wait(w)

        # restore frame
        self.play(Restore(frame), run_time=2)
        self.wait()

        # dot product
        dot0 = Tex(r"If $\mathbf{a} \perp \mathbf{b}$, then $\mathbf{a}\cdot \mathbf{b}=0$", font_size=65).to_corner(UL)
        self.play(Write(dot0))
        self.wait(w)

        # normal equation
        ne = MathTex(r"\mathbf{x} \cdot (\mathbf{v}-\mathbf{p})","=","0", font_size=70).next_to(dot0,DOWN).to_edge(LEFT)
        color_tex_standard(ne)
        self.play(TransformFromCopy(xl,ne[0][0]), run_time=1.75)
        self.play(Write(ne[0][1]))
        self.play(
            Write(ne[0][2]), Write(ne[0][-1]), # parentheses
            TransformFromCopy(rl[:],ne[0][3:6])
        , run_time=1.75)
        self.play(Write(ne[1]))
        self.play(Write(ne[2]))
        self.wait(w)

        # move to center-ish, remove dot product rule
        self.play(
            ne.animate.center().shift(LEFT*xcoords[0]),
            FadeOut(dot0)
        )
        self.wait()

        # distribute
        ne2 = MathTex(r"\mathbf{x}\cdot \mathbf{v} - \mathbf{x} \cdot \mathbf{p} = 0", font_size=70).shift(LEFT*xcoords[0])
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
        ne3 = MathTex(r"\mathbf{x}\cdot \mathbf{v} = \mathbf{x} \cdot \mathbf{p}", font_size=70).shift(LEFT*xcoords[0])
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
        ne4 = MathTex(r"\mathbf{x}\cdot \mathbf{p} = \mathbf{x} \cdot \mathbf{v}", font_size=70).shift(LEFT*xcoords[0])
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
        pxe = MathTex(r"\mathbf{p}=p_x \mathbf{x}", font_size=70).next_to(ne4,DOWN)
        pxe.shift(RIGHT*(ne4[0][3].get_center()[0]-pxe[0][1].get_center()[0]))
        color_tex_standard(pxe)
        self.play(FadeIn(pxe,shift=LEFT))
        self.wait()

        # substitute in
        ne5 = MathTex(r"\mathbf{x}\cdot p_x \mathbf{x} = \mathbf{x} \cdot \mathbf{v}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne5)
        self.play(*TransformBuilder(
            [ne4[0],pxe[0]],ne5,
            [
                ([0,[0,1]],[0,[0,1]]), # x dot
                ([0,2],None), # p
                ([1,slice(2,5)],[0,slice(2,5)]), # px x
                ([1,[0,1]],None), # p=
                ([0,slice(3,None)],[0,slice(5,None)]), # = onward
            ]
        ), run_time=1.5)
        self.wait(w)

        # solve for px
        ne6 = MathTex(r"p_x = \frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne6)
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

        # mag squared rule
        mag2 = MathTex(r"\mathbf{a} \cdot \mathbf{a} = |\mathbf{a}|^2", font_size=65).to_corner(UL)
        self.play(Write(mag2))
        self.wait(w)

        ne7 = MathTex(r"p_x = \frac{\mathbf{x} \cdot \mathbf{v}}{|\mathbf{x}|^2}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne7)
        AlignBaseline(ne7,ne6)
        self.play(*TransformBuilder(
            ne6[0],ne7[0],
            [
                (slice(0,7),slice(0,7)), # everything before denominator
                ([[7,9]],8,Merge), # x
                (None,7,FadeIn,{"shift":RIGHT}), # |
                (None,[[9,10]],FadeIn,{"shift":LEFT}), # |2
                (8,None), # dot
            ]
        ))
        self.wait(w)

        # case 1 unit vector
        c1 = Tex("Case 1: ",r"$|\mathbf{x}|=1$", font_size=70).next_to(ne7,DOWN,aligned_edge=LEFT)        
        color_tex(c1,r"$\mathbf{x}$",XCOLOR)
        self.play(Write(c1[0]))
        self.play(Write(c1[1]))
        self.wait(w)

        # sub in 1
        ne8 = MathTex(r"p_x = \frac{\mathbf{x} \cdot \mathbf{v}}{1}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne8)
        AlignBaseline(ne8,ne7)
        self.play(*TransformBuilder(
            [ne7[0],c1[1]],ne8,
            [
                ([0,slice(None,7)],[0,slice(None,7)]), # everything before denominator
                ([0,slice(7,None)],None), # denominator
                ([1,-1],[0,7], TransformFromCopy) # 1
            ]
        ), run_time=1.5)
        self.wait(w)

        # move case 1 to diagram
        self.play(c1.animate.next_to(VGroup(*[mob for mob in diagram if mob != axes]),DOWN,aligned_edge=RIGHT).shift(DOWN*0.2),run_time=1.75)

        # get rid of dot rule and fraction
        ne9 = MathTex(r"p_x = \mathbf{x} \cdot \mathbf{v}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne9)
        AlignBaseline(ne9,ne8)
        self.play(
            FadeOut(mag2),
            *TransformBuilder(
                ne8[0],ne9[0],
                [
                    (slice(None,3),slice(None,3)), # up to equals
                    (slice(3,6),slice(3,6)), # numerator
                    (slice(6,None),None), # denominator
                ]
            ), run_time=1.5
        )
        self.wait(w)

        # p formula
        pfe = MathTex(r"\mathbf{p} = p_x \mathbf{x}", font_size=70)
        color_tex_standard(pfe)
        pfe.shift(ne9[0][2].get_center() - pfe[0][1].get_center()).shift(UP)
        # pfe.shift(UP*(ne9[0][0].get_center()[1]-c1[0][0].get_center()[1]))
        self.play(Write(pfe))
        self.wait(w)

        # substitute in
        pfe2 = MathTex(r"\mathbf{p} =(\mathbf{x}\cdot \mathbf{v}) \mathbf{x}", font_size=70)
        color_tex_standard(pfe2)
        pfe2.shift(pfe[0][0].get_center() - pfe2[0][0].get_center())
        self.play(
            ReplacementTransform(pfe[0][0:2],pfe2[0][0:2]),
            FadeOut(pfe[0][2:4]),
            FadeIn(pfe2[0][2]), FadeIn(pfe2[0][6]),
            TransformFromCopy(ne9[0][3:6],pfe2[0][3:6]),
            ReplacementTransform(pfe[0][-1],pfe2[0][-1], path_arc=-120*DEGREES)
        , run_time=1.5)
        self.wait(w)

        # dot as transpose rule
        dast = MathTex(r"\mathbf{a}\cdot \mathbf{b}","=",r"\mathbf{a}^T \mathbf{b}", font_size=65).to_corner(UL)
        for i in [0,1,2]:
            self.play(Write(dast[i]))
        self.wait(w)

        # convert px
        ne10 = MathTex(r"p_x = \mathbf{x}^T \mathbf{v}", font_size=70).shift(LEFT*xcoords[0])
        color_tex_standard(ne10)
        AlignBaseline(ne10,ne9)
        self.play(ReplacementTransform(ne9,ne10), run_time=1.5)
        self.play(Indicate(ne10[0][4]))
        self.wait(w)

        # convert p
        pfe3 = MathTex(r"\mathbf{p} = \mathbf{x}^T \mathbf{v} \mathbf{x}", font_size=70)
        color_tex_standard(pfe3)
        pfe3.shift(pfe2[0][0].get_center() - pfe3[0][0].get_center())
        self.play(*TransformBuilder(
            pfe2[0],pfe3[0],
            [
                (slice(0,2), slice(0,2)), #p=
                (2,None), (6,None), # parentheses
                (3,2), # x
                (4,3), # dot to transpose
                (5,4), # v
                (7,5), # x
            ]
        ), run_time=1.5)
        self.play(Indicate(pfe3[0][3]))
        self.wait(w)

        # rearrange pfe
        pfe4 = MathTex(r"\mathbf{p} =\mathbf{x} \mathbf{x}^T \mathbf{v}", font_size=70)
        color_tex_standard(pfe4)
        pfe4.shift(pfe3[0][0].get_center() - pfe4[0][0].get_center())
        self.play(*TransformBuilder(
            pfe3[0],pfe4[0],
            [
                (slice(0,2), slice(0,2)), #p=
                (-1,2, None,{"path_arc":240*DEGREES}), # x
                (slice(2,4), slice(3,5)), # xt
                (4,5)
            ]
            ),
            FadeOut(dast)
            , run_time=1.5
        )
        self.wait(w)

        # projection matrix        
        el = Ellipse(width=1.5,height=0.75).move_to(pfe4[0][2:5]).rotate(20*DEGREES)
        pm = Tex("Projection Matrix", font_size=75).next_to(el,UP)
        self.play(
            DrawBorderThenFill(pm),
            Write(el)
        )
        self.wait(w)
        
        # projection matrix out
        self.play(
            Unwrite(el),
            DrawBorderThenFill(pm, reverse_rate_function=True)
        )
        self.wait(w)

        # inner and outer product
        inner = MathTex(r"\mathbf{a}\cdot \mathbf{b}=\mathbf{a}^T \mathbf{b}",r"\longleftarrow",r"\text{ Inner Product}", font_size=65).to_corner(UL)
        outer = MathTex(r"\mathbf{a}\otimes \mathbf{b}","=",r"\mathbf{a} \mathbf{b}^T",r"\longleftarrow",r"\text{ Outer Product}", font_size=65).next_to(inner,DOWN,aligned_edge=LEFT)
        for part in inner:
            self.play(Write(part))
        self.play(
            Write(outer[0]),
            VGroup(pfe4, ne10).animate.shift(DOWN),
            inner.animate.shift(RIGHT*(outer[1].get_center()[0]-inner[0][3].get_center()[0]))
        )
        for part in outer[1:]:
            self.play(Write(part))        
        
        self.wait(w)

        # down just to xxt
        self.play(
            FadeOut(inner,outer, shift=UP),
            FadeOut(v,p,x,r,vl,pl,xl,rl,ra,pfe4[0][-1], shift=RIGHT),
            FadeOut(pfe4[0][0:2], shift=LEFT),
            FadeOut(c1,ne10, shift=DOWN),
            run_time=1.5
        )
        xxt = MathTex(r"\mathbf{x} \mathbf{x}^T", font_size=80).to_edge(UP)
        color_tex_standard(xxt)
        self.play(ReplacementTransform(pfe4[0][2:5], xxt[0]),run_time=1.5)
        self.wait(w)




class xxTmatrix(Scene):
    def construct(self):
        def color_tex_matrix(matrix):
            for elem in matrix.get_entries():
                color_tex(elem, ("x",XCOLOR),("_1",XKCD.LIGHTRED),("_2",XKCD.LIGHTGREEN),("_3",XKCD.LIGHTBLUE))

        # back to previous scene
        xxt = MathTex(r"\mathbf{x} \mathbf{x}^T", font_size=80).to_edge(UP)
        color_tex_standard(xxt)
        self.add(xxt)
        self.wait(w)

        # from symbol to matrix
        x = Matrix([["x_1"],["x_2"],["x_3"]]).scale(1.1)
        xt = Matrix([["x_1","x_2","x_3"]]).scale(1.1)
        color_tex_matrix(x), color_tex_matrix(xt)                
        VGroup(x,xt).arrange()
        self.play(TransformFromCopy(xxt[0][0], x),run_time=1.5)
        self.play(TransformFromCopy(xxt[0][1:], xt),run_time=1.5)
        self.wait(w)

        # product matrix
        xxtm = Matrix([
            ["x_1 x_1","x_1 x_2", "x_1 x_3"],
            ["x_2 x_1","x_2 x_2", "x_2 x_3"],
            ["x_3 x_1","x_3 x_2", "x_3 x_3"]
        ]).scale(1.1)
        color_tex_matrix(xxtm)        
        x.generate_target().next_to(xxtm,LEFT), xt.generate_target().next_to(xxtm,UP)
        VGroup(x.target, xt.target, xxtm).center()
        self.play(
            MoveToTarget(x),
            MoveToTarget(xt)
        )        

        # add braces
        self.play(
            FadeIn(xxtm.get_brackets()[0],shift=LEFT),
            FadeIn(xxtm.get_brackets()[1],shift=RIGHT)
        )

        # multiply
        from itertools import product
        for (xi,xj), xij in zip(product(x[0],xt[0]),xxtm.get_entries()):
            self.play(
                Indicate(xi),
                Indicate(xj),
                TransformFromCopy(xi, xij[0][:2]),
                TransformFromCopy(xj, xij[0][2:])
            )
        self.wait(w)

        # formula next to matrix
        eq = MathTex("=", font_size=75).next_to(xxtm,LEFT)
        AlignBaseline(xxt.generate_target().next_to(eq,LEFT), eq)
        xxtm.generate_target()
        VGroup(xxt.target,eq,xxtm.target).center()
        self.play(
            MoveToTarget(xxt),
            MoveToTarget(xxtm),
            Write(eq),
            FadeOut(x,xt)
        , run_time=1.5)
        self.wait(w)

        # fade out everything
        self.play(FadeOut(*self.mobjects))





class NonUnit1d(MovingCameraScene):
    def construct(self):
        xcoords = np.array([2,1])
        vcoords = np.array([0.5,2])
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        # initial frame stuff
        frame = self.camera.frame
        frame.save_state()

        # vectors and labels, recover from earlier scene
        axes = Axes(x_range=[-2,2], x_length=4,y_range=[-2,2],y_length=4)
        x = Arrow(axes.c2p(0,0), axes.c2p(*xcoords), buff=0, color=XCOLOR)
        x2 = Arrow(axes.c2p(0,0), axes.c2p(*(xcoords*1.6)), buff=0, color=XCOLOR)
        v = Arrow(axes.c2p(0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR)
        xl = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x.get_tip(), RIGHT)
        x2l = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x2.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r).shift(UP*0.3+LEFT*0.4)
        color_tex_standard(rl)
        p = Arrow(axes.c2p(0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR)        
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DOWN)        
        ra = RightAngle(p,r,length=0.2,quadrant=(-1,1))
        diagram0 = VGroup(x,v,xl,vl,r,rl,p,pl,ra) 
        diagram = VGroup(x,x2,v,xl,x2l,vl,r,rl,p,pl,ra)
        VGroup(axes,diagram).to_corner(UR)
        
        # start zoomed, add diagram
        frame.scale(0.5).move_to(diagram0.get_center())
        self.play(FadeIn(diagram0, shift=UP))
        self.wait(w)

        # extend x
        x2 = Arrow(axes.c2p(0,0), axes.c2p(*(xcoords*1.6)), buff=0, color=XCOLOR)
        self.play(
            ReplacementTransform(x,x2),
            ReplacementTransform(xl,x2l)
        )
        self.wait(w)

        # zoom out for case
        c2 = MathTex(r"\text{Case 2: }",r"| \mathbf{x} |",r"\neq", "1", font_size=70).next_to(diagram,DOWN).to_edge(RIGHT)
        color_tex(c2,r"\mathbf{x}",XCOLOR)
        self.play(frame.animate.scale(1.2).move_to(VGroup(diagram,c2).get_center()))

        # case 2 caption        
        for part in c2: self.play(Write(part))
        self.wait(w)

        # zoom out for equations
        self.play(Restore(frame), run_time=2)
        self.wait(w)
        
        # formula back
        coef = MathTex(r"p_x =", r"\frac{\mathbf{x} \cdot \mathbf{v}}{|\mathbf{x}|^2}", font_size=70).shift(LEFT*xcoords[0]*1.4)
        color_tex_standard(coef)
        self.play(FadeIn(coef,shift=RIGHT))
        self.wait()

        # full equation for p        
        pfe = MathTex(r"\mathbf{p} =", r"\frac{\mathbf{x} \cdot \mathbf{v}}{|\mathbf{x}|^2}", r"\mathbf{x}", font_size=70).next_to(coef,UP).shift(DOWN*0.5)
        pfe.shift(RIGHT*(coef[0][2].get_center()[0]-pfe[0][1].get_center()[0]))
        color_tex_standard(pfe)
        pfe0 = MathTex(r"\mathbf{p} =", "p_x", r"\mathbf{x}", font_size=70).move_to(pfe,aligned_edge=LEFT)
        color_tex_standard(pfe0)
        self.play(coef.animate.shift(DOWN))
        self.play(FadeIn(pfe0,shift=RIGHT))        
        self.wait(w)

        # subsitute
        self.play(
            ReplacementTransform(pfe0[0],pfe[0]),
            ReplacementTransform(pfe0[-1],pfe[-1]),
            # Merge([coef[1].copy(),pfe0[1]], pfe[1])
            FadeOut(pfe0[1]), TransformFromCopy(coef[1],pfe[1]),
            run_time=1.75
        )
        self.wait(w)

        # indicate x's to cancel
        self.play(Indicate(pfe[1][0]))
        self.wait()
        self.play(Indicate(pfe[-1]))
        self.wait()
        self.play(Indicate(pfe[1][7]))
        self.wait(w)

        # to vector
        pfe1 = MathTex(r"\mathbf{p} =", r"\frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", r"\mathbf{x}", font_size=70).move_to(pfe,aligned_edge=LEFT)        
        color_tex_standard(pfe1)
        AlignBaseline(pfe1, pfe)
        self.play(
            *TransformBuilder(
                pfe, pfe1,
                [
                    (0,0), # lhs
                    ([1,slice(0,4)],[1,slice(0,4)]), # top
                    ([1,4],None,FadeOut,{"shift":LEFT}), # left |
                    ([1,[6,7]],None,FadeOut,{"shift":RIGHT}), # right |2
                    ([1,5],[1,4]), # first x
                    ([1,5],[1,6], TransformFromCopy), # second x
                    (None,[1,5]), # dot
                    (2,2) # x
                ]
            ), run_time=1.5
        )
        self.wait()

        # coefficient to dot product
        coef1 = MathTex(r"p_x =", r"\frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", font_size=70).move_to(coef,aligned_edge=LEFT)        
        color_tex_standard(coef1)
        AlignBaseline(coef1, coef)
        self.play(
            *TransformBuilder(
                coef, coef1,
                [
                    (0,0), # lhs
                    ([1,slice(0,4)],[1,slice(0,4)]), # top
                    ([1,4],None,FadeOut,{"shift":LEFT}), # left |
                    ([1,[6,7]],None,FadeOut,{"shift":RIGHT}), # right |2
                    ([1,5],[1,4]), # first x
                    ([1,5],[1,6], TransformFromCopy), # second x
                    (None,[1,5]), # dot
                ]
            ),run_time=1.5
        )
        self.wait(w)

        # component to matrix notation
        coef15 = MathTex(r"p_x =", r"\frac{\mathbf{x}^T \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", font_size=70).move_to(coef,aligned_edge=LEFT)        
        color_tex_standard(coef15)
        AlignBaseline(coef15, coef1)
        self.play(pfe1.animate.set_opacity(0.4))
        self.play(ReplacementTransform(coef1,coef15))
        self.wait()
        coef2 = MathTex(r"p_x =", r"\frac{\mathbf{x}^T \mathbf{v}}{\mathbf{x}^T \mathbf{x}}", font_size=70).move_to(coef,aligned_edge=LEFT)        
        color_tex_standard(coef2)
        AlignBaseline(coef2, coef15)
        self.play(ReplacementTransform(coef15,coef2))
        self.wait(w)

        # pull out denominator
        coef3 = MathTex(r"p_x =", r"\frac{1}{\mathbf{x}^T \mathbf{x}}",r"\mathbf{x}^T \mathbf{v}", font_size=70).move_to(coef2,aligned_edge=LEFT)        
        color_tex_standard(coef3)
        AlignBaseline(coef3, coef2)
        self.play(*TransformBuilder(
            coef2,coef3,
            [
                (0,0), # lhs
                ([1,slice(0,3)],[2,slice(0,3)]), # xtv
                (None,[1,0]), # 1
                ([1,3],[1,1]), # fraction
                ([1,slice(4,7)],[1,slice(2,5)]), # xtx                
            ]
        ),run_time=1.5)
        self.wait(w)

        # plug in to p
        pfe2 = MathTex(r"\mathbf{p} =", r"\frac{1}{\mathbf{x}^T \mathbf{x}}",r"\mathbf{x}^T \mathbf{v}", r"\mathbf{x}", font_size=70).move_to(pfe,aligned_edge=LEFT)        
        color_tex_standard(pfe2)
        AlignBaseline(pfe2, pfe1)
        self.play(pfe1.animate.set_opacity(1))
        self.play(
            ReplacementTransform(pfe1[0],pfe2[0]),            
            FadeOut(pfe1[1]),
            TransformFromCopy(coef3[1],pfe2[1]),
            TransformFromCopy(coef3[2],pfe2[2]),
            ReplacementTransform(pfe1[2],pfe2[3])
        , run_time=2)
        self.wait(w)

        # rearrange p
        pfe3 = MathTex(r"\mathbf{p} =", r"\frac{1}{\mathbf{x}^T \mathbf{x}}", r"\mathbf{x}",r"\mathbf{x}^T \mathbf{v}", font_size=70).move_to(pfe,aligned_edge=LEFT)        
        color_tex_standard(pfe3)
        AlignBaseline(pfe3, pfe2)
        self.play(*TransformBuilder(
            pfe2,pfe3,
            [
                (0,0), # lhs
                (1,1), # fraction
                (3,2,None,{"path_arc":240*DEGREES}), # x
                (2,3), # xtv
            ]
        ),run_time=1.5)
        self.wait(w)

        # projection matrix
        el = Ellipse(width=3.25,height=1.75).move_to(pfe3[1:3]).rotate(10*DEGREES).shift(RIGHT*0.3)
        pm = Tex("Projection Matrix", font_size=75).next_to(el,UP)
        self.play(
            DrawBorderThenFill(pm),
            Write(el)
        )
        self.wait(w)

        # outer product
        el2 = Ellipse(width=1.5,height=0.75).move_to(VGroup(pfe3[2],pfe3[3][0:2])).rotate(20*DEGREES)
        op = Tex("Outer", " Product", font_size=75).move_to(pm)
        self.play(
            ReplacementTransform(el,el2),
            ReplacementTransform(pm,op)
        )
        self.wait(w)

        # inner product
        el3 = Ellipse(width=1.6,height=0.85).move_to(pfe3[1][2:])
        ip = Tex("Inner", " Product", font_size=75).move_to(op)
        self.play(
            ReplacementTransform(el2,el3),
            ReplacementTransform(op[1],ip[1]),
            FadeOut(op[0],shift=DOWN), FadeIn(ip[0],shift=DOWN)
        )
        self.wait(w)

        # remove ellipse,label
        self.play(FadeOut(el3,ip))
        self.wait(w)

        # rearrange once more
        pfe4 = MathTex(r"\mathbf{p} =",r"\mathbf{x}", r"\frac{1}{\mathbf{x}^T \mathbf{x}}", r"\mathbf{x}^T \mathbf{v}", font_size=70).move_to(pfe,aligned_edge=LEFT)        
        color_tex_standard(pfe4)
        AlignBaseline(pfe4, pfe3)
        self.play(*TransformBuilder(
            pfe3,pfe4,
            [
                (0,0), # lhs
                (2,1,None,{"path_arc":240*DEGREES}), # x
                (1,2), # fraction
                (3,3), # xtv
            ]
        ),run_time=1.5)
        self.wait(w)

        # get rid of stuff, then zoom in
        self.play(
            FadeOut(c2, shift=RIGHT),
            FadeOut(pfe4, coef3,shift=LEFT)
        )
        self.play(frame.animate.scale(0.5).move_to(diagram), run_time=2)
        self.wait(w)



class Transition2d(ThreeDScene):
    def construct(self):
        xcoords = np.array([2,1,0])
        vcoords = np.array([0.5,2,0])
        pcoords = xcoords * np.dot(xcoords,vcoords) / np.dot(xcoords,xcoords)
        
        

        # vectors and labels, recover from earlier scene
        axes = ThreeDAxes(
            x_range=[-2,2], x_length=4,
            y_range=[-2,2],y_length=4,
            z_range=[-2,2],z_length=4).set_opacity(0)
        x2 = Arrow(axes.c2p(0,0,0), axes.c2p(*(xcoords*1.6)), buff=0, color=XCOLOR,stroke_width=12)
        v = Arrow(axes.c2p(0,0,0), axes.c2p(*vcoords), buff=0, color=VCOLOR,stroke_width=12)        
        r = Arrow(axes.c2p(*pcoords), axes.c2p(*vcoords), buff=0,stroke_width=12)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
        p = Arrow(axes.c2p(0,0,0), axes.c2p(*pcoords), buff=0, color=PCOLOR,stroke_width=12)        
        
        
        x2l = MathTex(r"\mathbf{x}", font_size=60, color=XCOLOR).next_to(x2.get_tip(), RIGHT)
        vl = MathTex(r"\mathbf{v}", font_size=60, color=VCOLOR).next_to(v.get_tip(), UP)        
        rl = MathTex(r"\mathbf{v}-\mathbf{p}", font_size=60).next_to(r).shift(UP*0.3+LEFT*0.4)
        color_tex_standard(rl)
        pl = MathTex(r"\mathbf{p}", font_size=60, color=PCOLOR).next_to(p.get_tip(), DOWN)        
        
        ra = RightAngle(p,r,length=0.2,quadrant=(-1,1))
        
        plane = Surface(lambda u,v: axes.c2p(u,u*xcoords[1]/xcoords[0],v),u_range=[-0.422,4],v_range=[-3,5],resolution=1).set_opacity(0)        
        
        diagram = VGroup(axes,plane,x2,v,x2l,vl,r,rl,p,pl,ra)
        diagram.shift(-VGroup(x2,v,x2l,vl,r,rl,p,pl,ra).get_center())
                
    
        # start zoomed, add diagram                
        self.add(diagram)
        self.set_camera_orientation(zoom=2)
        self.wait()        
        
        anims = [
            diagram.animate.rotate(-26*DEGREES,axis=OUT).rotate(16*DEGREES,axis=RIGHT),
            plane  .animate.rotate(-26*DEGREES,axis=OUT).rotate(16*DEGREES,axis=RIGHT).set_opacity(0.5)
        ]
        self.move_camera(zoom=1.9,added_anims=anims,run_time=2)        
        self.wait(w)

        # fade out, to question mark                        
        qm = Tex("?", font_size=500).set_color(XKCD.PINKYRED)
        self.add_fixed_in_frame_mobjects(qm)        
        self.play(
            FadeOut(diagram),
            FadeIn(qm)
        )
        self.wait(w)

        # fade out qm
        self.play(FadeOut(qm))




class Project2dOrthonormal(MovingCameraScene):
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
        ra = VGroup(
            Line(axes @ (0.9*pcoords),axes @ (0.9*pcoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*pcoords+OUT*0.1),axes @ (1*pcoords+OUT*0.1), stroke_width=2)
        )
        vectors = VGroup(v,x,y,p,px,py,r)
        dashes = VGroup(dy,dx)

        plane = Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=1).set_opacity(0.5)
        
        diagram = VGroup(axes,plane,vectors,dashes, ra)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UP,buff=0.25)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UR,buff=0.15)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl)
        diagram.add(labels)        
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.5+RIGHT*0.2)
        self.camera.frame.scale(0.4) # self.set_camera_orientation(zoom=2)
        for vector in (vector for vector in vectors): face_camera(self,vector)
        
        # add v        
        self.play(FadeIn(v))
        self.play(Write(vl))
        self.play(Create(plane),run_time=2)
        self.wait(w)

        # add x, y
        self.play(GrowArrow(x))
        self.play(Write(xl))
        self.play(GrowArrow(y))
        self.play(Write(yl))
        self.wait(w)

        # add p
        self.play(TransformFromCopy(v,p),run_time=1.5)
        self.play(Write(pl))
        self.wait(w)
        
        # to px, py
        self.play(
            TransformFromCopy(p,px),
            Create(dy)
        , run_time=1.75)
        self.play(Write(pxl))
        self.play(
            TransformFromCopy(p,py),
            Create(dx)
        , run_time=1.75)
        self.play(Write(pyl))
        self.wait(w)

        # zoom out, write equation below
        pe = MathTex(r"\mathbf{p}","=",r"p_x \mathbf{x}","+",r"p_y \mathbf{y}", font_size=60).next_to(diagram,DOWN)
        color_tex_standard(pe)
        pe[0].set_color(PCOLOR), pe[2][-1].set_color(XCOLOR),pe[4][0:2].set_color(PCOLOR), pe[4][-1].set_color(YCOLOR)
        self.play(self.camera.frame.animate.scale(1.2).move_to(VGroup(*[mob for mob in diagram if mob != axes],pe)),run_time=1.25)
        self.play(TransformFromCopy(pl[0],pe[0]),run_time=1.5)
        self.play(Write(pe[1]))
        self.play(
            TransformFromCopy(pxl[0],pe[2]),
            Indicate(pxl[0]),
            run_time=1.5
        )
        self.play(Write(pe[3]))
        self.play(
            TransformFromCopy(pyl[0],pe[4]),
            Indicate(pyl[0]),
            run_time=1.5
        )
        self.wait(w)

        # rejection vector
        self.play(TransformFromCopy(v,r), run_time=1.75)
        self.play(Write(rl))
        self.wait(w)

        # perpendicular
        self.play(Write(ra))
        self.wait(w)

        # zoom out
        self.play(
            self.camera.frame.animate.scale(1/(0.4*1.2)),
            diagram.animate.to_corner(UR),            
            pe.animate.next_to(diagram.copy().to_corner(UR),DOWN)
        , run_time=2.5)
        self.wait(w)

        # to normal equations
        nex = MathTex(r"(\mathbf{v}-\mathbf{p})\cdot \mathbf{x} =0", font_size=65).shift(3*LEFT+UP*1.5)
        color_tex_standard(nex)
        ney = MathTex(r"(\mathbf{v}-\mathbf{p})\cdot \mathbf{y} =0", font_size=65).shift(3*LEFT+DOWN*1.5)
        color_tex_standard(ney)
        self.play(
            Write(nex[0][0]), Write(nex[0][4]), Write(ney[0][0]), Write(ney[0][4]), # parentheses
            ReplacementTransform(rl[0].copy(),nex[0][1:4]), ReplacementTransform(rl[0].copy(),ney[0][1:4]) # v-p
        , run_time=2)
        self.play(Write(nex[0][5]), Write(ney[0][5])) # dot
        self.play(ReplacementTransform(xl[0].copy(),nex[0][6]), ReplacementTransform(yl[0].copy(),ney[0][6]), run_time=2) # x,y
        self.play(Write(nex[0][-2:]), Write(ney[0][-2:]),run_time=1.5) # =0        
        self.wait(w)

        # rearrange normal equations
        nex1 = MathTex(r"\mathbf{x} \cdot \mathbf{p}","=",r"\mathbf{x} \cdot \mathbf{v}", font_size=65).move_to(nex)
        AlignBaseline(nex1,nex)
        color_tex_standard(nex1)
        ney1 = MathTex(r"\mathbf{y} \cdot \mathbf{p}","=",r"\mathbf{y} \cdot \mathbf{v}", font_size=65).move_to(ney)
        AlignBaseline(ney1,ney)
        color_tex_standard(ney1)
        self.play(*TransformBuilder(
            nex,nex1,
            [
                ([0,[0,4]],None), # ()
                ([0,1],[-1,-1],None,{"path_arc":120*DEGREES}), # v
                ([0,2],None), # -
                ([0,3],[0,2]), # p
                ([0,5],[0,1],None,{"path_arc":120*DEGREES}), # dot
                ([0,5],[2,1], TransformFromCopy,{"path_arc":-120*DEGREES}), #dot
                ([0,6],[0,0],None,{"path_arc":120*DEGREES}), # x
                ([0,6],[2,0],TransformFromCopy), # x
                ([0,7],[1,0]), # =
                ([0,-1],None) # 0
            ]
        ), run_time=2.5)
        self.play(*TransformBuilder(
            ney,ney1,
            [
                ([0,[0,4]],None), # ()
                ([0,1],[-1,-1],None,{"path_arc":120*DEGREES}), # v
                ([0,2],None), # -
                ([0,3],[0,2]), # p
                ([0,5],[0,1],None,{"path_arc":120*DEGREES}), # dot
                ([0,5],[2,1], TransformFromCopy,{"path_arc":-120*DEGREES}), #dot
                ([0,6],[0,0],None,{"path_arc":120*DEGREES}), # y
                ([0,6],[2,0],TransformFromCopy), # y
                ([0,7],[1,0]), # =
                ([0,-1],None) # 0
            ]
        ), run_time=2.5)
        self.wait(w)

        # substitute p
        nex2 = MathTex(r"\mathbf{x} \cdot (p_x \mathbf{x} + p_y \mathbf{y})","=",r"\mathbf{x} \cdot \mathbf{v}", font_size=65).move_to(nex1)        
        color_tex_standard(nex2)
        AlignBaseline(nex2,nex1)
        ney2 = MathTex(r"\mathbf{y} \cdot (p_x \mathbf{x} + p_y \mathbf{y})","=",r"\mathbf{y} \cdot \mathbf{v}", font_size=65).move_to(ney1)        
        color_tex_standard(ney2)
        AlignBaseline(ney2,ney1)
        self.play(
            ReplacementTransform(nex1[0][0], nex2[0][0]), # x
            ReplacementTransform(nex1[0][1], nex2[0][1]), # dot
            FadeIn(nex2[0][2]), FadeIn(nex2[0][-1]), # ()
            FadeOut(nex1[0][2]), # p
            TransformFromCopy(pe[2][:],nex2[0][3:6],path_arc=60*DEGREES), # px x
            TransformFromCopy(pe[3][0], nex2[0][6],path_arc=60*DEGREES), # +
            TransformFromCopy(pe[4][:],nex2[0][7:10],path_arc=60*DEGREES), # py y
            ReplacementTransform(nex1[1], nex2[1]), # =
            ReplacementTransform(nex1[2], nex2[2]) # rhs
        , run_time=2)
        self.wait()        
        self.play(
            ReplacementTransform(ney1[0][0], ney2[0][0]), # x
            ReplacementTransform(ney1[0][1], ney2[0][1]), # dot
            FadeIn(ney2[0][2]), FadeIn(ney2[0][-1]), # ()
            FadeOut(ney1[0][2]), # p
            TransformFromCopy(pe[2][:],ney2[0][3:6],path_arc=-60*DEGREES), # px x
            TransformFromCopy(pe[3][0], ney2[0][6],path_arc=-60*DEGREES), # +
            TransformFromCopy(pe[4][:],ney2[0][7:10],path_arc=-60*DEGREES), # py y
            ReplacementTransform(ney1[1], ney2[1]), # =
            ReplacementTransform(ney1[2], ney2[2]) # rhs
        , run_time=2)
        self.wait()

        # distribute dot products
        nex3 = MathTex(r"p_x \mathbf{x} \cdot \mathbf{x} + p_y \mathbf{x} \cdot \mathbf{y}","=",r"\mathbf{x} \cdot \mathbf{v}", font_size=65).move_to(nex1)        
        color_tex_standard(nex3)
        AlignBaseline(nex3,nex1)
        ney3 = MathTex(r"p_x \mathbf{y} \cdot \mathbf{x} + p_y \mathbf{y} \cdot \mathbf{y}","=",r"\mathbf{y} \cdot \mathbf{v}", font_size=65).move_to(ney1)        
        color_tex_standard(ney3)
        AlignBaseline(ney3,ney1)
        self.play(*TransformBuilder(
            nex2,nex3,
            [
                ([0,0],[0,2],None,{"path_arc":-280*DEGREES}), # x
                ([0,0],[0,8],TransformFromCopy,{"path_arc":120*DEGREES}), # x
                ([0,1],[0,3],None,{"path_arc":-280*DEGREES}), # dot
                ([0,1],[0,9],TransformFromCopy,{"path_arc":120*DEGREES}), # dot
                ([0,2],None), ([0,-1],None), # ()
                ([0,[3,4]], [0,[0,1]]), # px
                ([0,5],[0,4]), # x
                ([0,6],[0,5]), # +
                ([0,[7,8]],[0,[6,7]]), #py
                ([0,9],[0,10]), # y
                (1,1), (2,2) # = rhs
            ]
        )
        ,run_time=2)
        self.play(*TransformBuilder(
            ney2,ney3,
            [
                ([0,0],[0,2],None,{"path_arc":-280*DEGREES}), # y
                ([0,0],[0,8],TransformFromCopy,{"path_arc":120*DEGREES}), # y
                ([0,1],[0,3],None,{"path_arc":-280*DEGREES}), # dot
                ([0,1],[0,9],TransformFromCopy,{"path_arc":120*DEGREES}), # dot
                ([0,2],None), ([0,-1],None), # ()
                ([0,[3,4]], [0,[0,1]]), # px
                ([0,5],[0,4]), # x
                ([0,6],[0,5]), # +
                ([0,[7,8]],[0,[6,7]]), #py
                ([0,9],[0,10]), # y
                (1,1), (2,2) # = rhs
            ]
        )
        ,run_time=2)
        self.wait(w)

        # c1 orthonormal
        c1 = MathTex(r"\text{Case 1: }", r"\mathbf{x}, \mathbf{y} \text{ Orthonormal}").move_to(pe)
        color_tex_standard(c1)
        self.play(pe.animate.shift(DOWN))
        self.play(Write(c1[0]))
        self.play(Write(c1[1]))
        ortho = MathTex(r"|a|=|b|=1,", r" a\perp b", r"\longleftarrow",r"\text{Orthonormal}", font_size=50).to_corner(UL)
        for part in ortho: self.play(Write(part))
        self.wait(w)

        # indicate cross terms
        self.play(
            Circumscribe(nex3[0][-3:],color=RCOLOR, fade_in=False,fade_out=True),
            Circumscribe(ney3[0][2:5],color=RCOLOR, fade_in=False,fade_out=True)
        , run_time=2.5)
        self.wait(w)

        # cross terms to zero
        nex4 = MathTex(r"p_x \mathbf{x} \cdot \mathbf{x}","=",r"\mathbf{x} \cdot \mathbf{v}", font_size=65).shift(3*LEFT+UP*1.5)        
        color_tex_standard(nex4)
        AlignBaseline(nex4,nex3)
        ney4 = MathTex(r"p_y \mathbf{y} \cdot \mathbf{y}","=",r"\mathbf{y} \cdot \mathbf{v}", font_size=65).shift(3*LEFT+DOWN*1.5)        
        color_tex_standard(ney4)
        AlignBaseline(ney4,ney3)
        self.play(*TransformBuilder(
            nex3,nex4,
            [
                ([0,slice(0,5)],[0,slice(0,5)]), # px xdotx
                ([0,slice(5,None)],None,ShrinkToCenter), # cross term
                (1,1), (2,2) # = rhs
            ]
        ))
        self.play(*TransformBuilder(
            ney3,ney4,
            [
                ([0,slice(0,6)],None,ShrinkToCenter), # px ydotx +
                ([0,slice(6,None)],[0,slice(0,None)]), # cross term
                (1,1), (2,2) # = rhs
            ]
        ))
        self.wait(w)
        
        # divide over squared terms
        nex5 = MathTex(r"p_x","=", r"\frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", font_size=65).move_to(nex4)
        color_tex_standard(nex5)
        AlignBaseline(nex5,nex4)
        ney5 = MathTex(r"p_y","=", r"\frac{\mathbf{y} \cdot \mathbf{v}}{\mathbf{y} \cdot \mathbf{y}}", font_size=65).move_to(ney4)
        color_tex_standard(ney5)
        AlignBaseline(ney5,ney4)
        self.play(
            *TransformBuilder(
                nex4,nex5,
                [
                    ([0,[0,1]],[0,[0,1]]), # px
                    ([0,slice(2,5)],[2,slice(4,None)],None,{"path_arc":90*DEGREES}), # xdotx
                    (1,1), # =
                    ([2,slice(None,3)],[2,slice(None,3)]), # xdotv
                    (None,[2,3],FadeIn), # fraction
                ]
            ),
            *TransformBuilder(
                ney4,ney5,
                [
                    ([0,[0,1]],[0,[0,1]]), # px
                    ([0,slice(2,5)],[2,slice(4,None)],None,{"path_arc":90*DEGREES}), # xdotx
                    (1,1), # =
                    ([2,slice(None,3)],[2,slice(None,3)]), # xdotv
                    (None,[2,3],FadeIn), # fraction
                ]
            ), run_time=1.5
        )
        self.wait(w)

        # normal, denominators to 1
        nex6 = MathTex(r"p_x","=", r"\frac{\mathbf{x} \cdot \mathbf{v}}{1}", font_size=65).move_to(nex5)
        color_tex_standard(nex6)
        AlignBaseline(nex6,nex5)
        ney6 = MathTex(r"p_y","=", r"\frac{\mathbf{y} \cdot \mathbf{v}}{1}", font_size=65).move_to(ney5)
        color_tex_standard(ney6)
        AlignBaseline(ney6,ney5)
        self.play(
            *TransformBuilder(
                nex5,nex6,
                [
                    (0,0), (1,1), # lhs=
                    ([2,slice(0,4)], [2,slice(0,4)]), # numerator
                    ([2,slice(4,None)], [2,4],Merge), # denom -> 1                    
                ]
            ),
            *TransformBuilder(
                ney5,ney6,
                [
                    (0,0), (1,1), # lhs=
                    ([2,slice(0,4)], [2,slice(0,4)]), # numerator
                    ([2,slice(4,None)], [2,4],Merge), # denom -> 1                    
                ]
            ),run_time=1.5
        )
        self.wait(w)

        # down to no fraction
        nex7 = MathTex(r"p_x","=", r"\mathbf{x} \cdot \mathbf{v}", font_size=65).move_to(nex6,aligned_edge=LEFT)
        color_tex_standard(nex7)
        AlignBaseline(nex7,nex6)
        ney7 = MathTex(r"p_x","=", r"\mathbf{x} \cdot \mathbf{v}", font_size=65).move_to(ney6,aligned_edge=LEFT)
        color_tex_standard(ney7)
        AlignBaseline(ney7,ney6)
        self.play(
            *TransformBuilder(
                nex6,nex7,
                [
                    (0,0), (1,1), # lhs=
                    ([2,slice(0,3)], [2,slice(0,3)]), # numerator
                    ([2,slice(3,None)],None) # denominator
                ]
            ),
            *TransformBuilder(
                ney6,ney7,
                [
                    (0,0), (1,1), # lhs=
                    ([2,slice(0,3)], [2,slice(0,3)]), # numerator
                    ([2,slice(3,None)],None) # denominator
                ]
            )
        )
        self.wait(w)

        # substitute in components for p
        pe2 = MathTex(r"\mathbf{p}","=",r"(\mathbf{x}\cdot \mathbf{v}) \mathbf{x}","+",r"(\mathbf{y}\cdot \mathbf{v}) \mathbf{y}", font_size=60).move_to(pe)
        color_tex_standard(pe2)
        AlignBaseline(pe2,pe)
        self.play(
            *TransformBuilder(
                pe,pe2,
                [
                    (0,0),(1,1), # p=
                    (3,3), # +
                    ([2,-1], [2,-1]), # x
                    ([4,-1], [4,-1]), # y
                    (None,[2,0]), (None,[2,4]), # ()
                    (None,[4,0]), (None,[4,4]), # ()
                    ([2,[0,1]],None), ([4,[0,1]],None), # px, py
                ]
            ),            
            TransformFromCopy(nex7[2][:],pe2[2][1:4], path_arc=-120*DEGREES),
            TransformFromCopy(ney7[2][:],pe2[4][1:4], path_arc=-120*DEGREES),
            FadeOut(ortho)
        , run_time=2)
        self.wait(w)

        # to matrix notation
        # substitute in components for p
        nex8 = MathTex(r"p_x","=", r"\mathbf{x}^T \mathbf{v}", font_size=65).move_to(nex7,aligned_edge=LEFT)
        color_tex_standard(nex8)
        AlignBaseline(nex8,nex7)
        ney8 = MathTex(r"p_y","=", r"\mathbf{y}^T \mathbf{v}", font_size=65).move_to(ney7,aligned_edge=LEFT)
        color_tex_standard(ney8)
        AlignBaseline(ney8,ney7)
        self.play(
            ReplacementTransform(nex7,nex8),
            ReplacementTransform(ney7,ney8)
        , run_time=2)
        pe3 = MathTex(r"\mathbf{p}","=",r"\mathbf{x}^T \mathbf{v} \mathbf{x}","+",r"\mathbf{y}^T \mathbf{v} \mathbf{y}", font_size=60).move_to(pe2)
        color_tex_standard(pe3)
        AlignBaseline(pe3,pe2)
        self.play(
            *TransformBuilder(
                pe2,pe3,
                [
                    (0,0),(1,1), # p=
                    ([2,0],None), ([2,4],None), ([4,0],None), ([4,4],None), # ()
                    ([2,1],[2,0]), ([2,2],[2,1]), ([2,3],[2,2]), # x dot v
                    ([2,-1],[2,-1]), ([4,-1],[4,-1]), # x,y
                    (3,3),
                    ([4,1],[4,0]), ([4,2],[4,1]), ([4,3],[4,2]), # y dot v
                ]
            ),
            run_time=2
        )
        self.remove(*self.mobjects)
        self.add(pe3,nex8,ney8,c1,diagram)
        pe4 = MathTex(r"\mathbf{p}","=",r"\mathbf{x} \mathbf{x}^T \mathbf{v}","+",r"\mathbf{y} \mathbf{y}^T \mathbf{v}", font_size=60).move_to(pe3)
        color_tex_standard(pe4)
        AlignBaseline(pe4,pe3)
        self.play(*TransformBuilder(
            pe3,pe4,
            [
                (0,0),(1,1), # p=
                ([2,-1],[2,0], None,{"path_arc":-220*DEGREES}), # x                     
                ([2,0],[2,1]),([2,1],[2,2]), ([2,2],[2,3]), # xtv
                (3,3),
                ([4,-1],[4,0], None,{"path_arc":-220*DEGREES}), # y
                ([4,0],[4,1]),([4,1],[4,2]), ([4,2],[4,3]), # xtv
            ]
        ), run_time=1.5)
        self.wait(w)

        # move py
        self.play(ney8.animate.next_to(nex8,DOWN,aligned_edge=LEFT))

        # make p bigger    
        self.remove(*self.mobjects)
        self.add(pe4,nex8,ney8,c1,diagram)           
        pe5 = MathTex(r"\mathbf{p}","=",r"\mathbf{x} \mathbf{x}^T \mathbf{v}","+",r"\mathbf{y} \mathbf{y}^T \mathbf{v}", font_size=60)
        color_tex_standard(pe5).scale(80/60).next_to(c1.copy().shift(LEFT*c1.get_center()[0]), DOWN)
        self.play(
            ReplacementTransform(pe4,pe5),
            FadeOut(nex8,ney8)
        ,run_time=1.75)
        self.wait(w)

        # factor out v
        pe6 = MathTex(r"\mathbf{p}","=",r"( \mathbf{x} \mathbf{x}^T ","+",r"\mathbf{y} \mathbf{y}^T  )",r"\mathbf{v}", font_size=80).move_to(pe5)
        color_tex_standard(pe6)
        AlignBaseline(pe6,pe5)
        self.play(*TransformBuilder(
            pe5,pe6,
            [
                (0,0),(1,1), # p=
                (None,[2,0]), (None,[4,-1]), # ()
                ([2,slice(0,3)],[2,slice(1,4)]), # xxt
                (3,3), # +
                ([4,slice(0,3)],[4,slice(0,3)]), # yyt
            ]
        ), Merge([pe5[2][3],pe5[4][3]],pe6[-1][0],{"path_arc":120*DEGREES})
        , run_time=2)
        self.wait(w)

        # to matrix equation
        peq = MathTex(r"\mathbf{p}","=", font_size=80).move_to(pe6,aligned_edge=LEFT)
        color_tex_standard(peq)
        AlignBaseline(peq,pe6)
        xy = Matrix([[r"\mathbf{x}",r"\mathbf{y}"]], element_alignment_corner=UL).next_to(peq)
        color_tex_standard(xy)
        xtyt = Matrix([
            [r"\mathbf{x}^T"],
            [r"\mathbf{y}^T"]
        ]).next_to(xy)
        color_tex_standard(xtyt)
        veq = MathTex(r"\mathbf{v}",font_size=80).next_to(xtyt)
        color_tex_standard(veq)
        VGroup(peq, xy,xtyt,veq).next_to(c1.copy().shift(LEFT*c1.get_center()[0]), DOWN)
        self.play(
            ReplacementTransform(pe6[0:2],peq[0:2]), # p=
            ReplacementTransform(pe6[2][0], xy[1]), # ( to [
            ReplacementTransform(pe6[3],VGroup(xy[2],xtyt[1])), # + to ][
            ReplacementTransform(pe6[4][-1], xtyt[2]), # ) to ]
            ReplacementTransform(pe6[2][1], xy[0][0][0], path_arc=120*DEGREES), # x
            ReplacementTransform(pe6[2][2], xtyt[0][0][0][0], path_arc=-120*DEGREES),ReplacementTransform(pe6[2][3], xtyt[0][0][0][1], path_arc=-120*DEGREES), # xt
            ReplacementTransform(pe6[4][0], xy[0][1][0], path_arc=-120*DEGREES), # y
            ReplacementTransform(pe6[4][1], xtyt[0][1][0][0], path_arc=-120*DEGREES),ReplacementTransform(pe6[4][2], xtyt[0][1][0][1], path_arc=-120*DEGREES), # yt
            ReplacementTransform(pe6[-1],veq[0])        
        , run_time=2.5)        
        self.wait(w)

        # expand x y
        xyex = Matrix([
            ["x_1","y_1"],
            ["x_2","y_2"],
            ["x_3","y_3"],
        ])
        for row in xyex.get_rows():
            row[0].set_color(XCOLOR)
            row[1].set_color(YCOLOR)           
        for mob in [peq,xtyt,veq]: mob.generate_target()
        VGroup(peq.target,xyex,xtyt.target,veq.target).arrange().next_to(c1.copy().shift(LEFT*c1.get_center()[0]), DOWN)
        AlignBaseline(veq.target,peq.target)
        self.play(
            ReplacementTransform(xy.get_brackets(),xyex.get_brackets()),
            ReplacementTransform(xy.get_entries()[0],xyex.get_columns()[0]),
            ReplacementTransform(xy.get_entries()[1],xyex.get_columns()[1]),
            *[MoveToTarget(mob) for mob in [peq,xtyt,veq]]
        , run_time=1.5)        

        # expand xt yt
        xtytex = Matrix([
            ["x_1","x_2","x_3"],
            ["y_1","y_2","y_3"]
        ])
        xtytex.get_rows()[0].set_color(XCOLOR), xtytex.get_rows()[1].set_color(YCOLOR)
        for mob in [peq,xyex,veq]: mob.generate_target()
        VGroup(peq.target,xyex.target,xtytex,veq.target).arrange().next_to(c1.copy().shift(LEFT*c1.get_center()[0]), DOWN)
        AlignBaseline(veq.target,peq.target)
        self.play(
            ReplacementTransform(xtyt.get_brackets(),xtytex.get_brackets()),
            ReplacementTransform(xtyt.get_entries()[0],xtytex.get_rows()[0]),
            ReplacementTransform(xtyt.get_entries()[1],xtytex.get_rows()[1]),
            *[MoveToTarget(mob) for mob in [peq,xyex,veq]]
        , run_time=1.5)
        self.wait(w)

        # matrix to A
        peqA = MathTex(r"\mathbf{p}","=","A", font_size=80)
        color_tex_standard(peqA)
        for mob in [xtytex,veq]: mob.generate_target()
        VGroup(peqA,xtytex.target,veq.target).arrange().move_to(VGroup(peq,xyex,xtytex,veq))
        AlignBaseline(veq.target,peqA)
        self.play(
            ReplacementTransform(peq[0:2],peqA[0:2]), # p=
            ReplacementTransform(xyex,peqA[2]), # matrix to A
            MoveToTarget(xtytex), MoveToTarget(veq) # matrix v
        ,run_time=1.5)
        
        # matrix to A transpose
        peqATv = MathTex(r"\mathbf{p}","=","A","A^T",r"\mathbf{v}", font_size=80)
        color_tex_standard(peqATv)
        peqATv.move_to(VGroup(peqA,xtytex,veq))
        self.play(
            ReplacementTransform(peqA[0:3],peqATv[0:3]), # p=A
            ReplacementTransform(xtytex,peqATv[3]), # matrix to At
            ReplacementTransform(veq,peqATv[4]) # v
        ,run_time=1.5)
        self.wait(w)

        # A for reference
        Aeq = MathTex("A","=", font_size=70)
        xy = Matrix([[r"\mathbf{x}",r"\mathbf{y}"]], 
            element_alignment_corner=UL,element_to_mobject_config={"font_size":65})
        color_tex_standard(xy)
        VGroup(Aeq,xy).arrange().next_to(peqATv,DOWN)
        xy.shift(UP*(Aeq[1].get_center()[1]-xy.get_center()[1]))
        self.play(TransformFromCopy(peqATv[2],Aeq[0])) # A
        self.play(Write(Aeq[1])) # =
        self.play(FadeIn(xy,shift=DR))
        self.wait(w)

        # get components back
        self.play(FadeIn(nex8,ney8))
        self.wait()

        # components to matrix
        comps = Matrix([
            ["p_x"],
            ["p_y"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65}).next_to(VGroup(nex8[1],ney8[1]),LEFT)
        color_tex_standard(comps)
        self.play(
            FadeIn(comps.get_brackets()),
            ReplacementTransform(nex8[0],comps.get_entries()[0][0]), #x
            ReplacementTransform(ney8[0],comps.get_entries()[1][0]) #y
        )
        eq = MathTex("=", font_size=65).next_to(comps,RIGHT)
        self.play(Merge([nex8[1],ney8[1]],eq))
        xtyt = Matrix([[r"\mathbf{x}^T"],[r"\mathbf{y}^T"]]).next_to(eq)
        color_tex_standard(xtyt)
        veq = MathTex(r"\mathbf{v}",font_size=65).next_to(xtyt)
        color_tex_standard(veq)
        self.play(
            ReplacementTransform(nex8[2][:2],xtyt.get_entries()[0][0][:]), #x
            ReplacementTransform(ney8[2][:2],xtyt.get_entries()[1][0][:]), #y
            FadeIn(xtyt.get_brackets()),
            Merge([nex8[2][2],ney8[2][2]],veq)
        ,run_time=1.5)
        self.wait(w)

        # combine
        atv = MathTex("=","A^T",r"\mathbf{v}",font_size=65)
        color_tex_standard(atv)
        atv.shift(eq.get_center()-atv[0].get_center())
        self.play(
            ReplacementTransform(eq,atv[0]), #=
            TransformFromCopy(Aeq[0],atv[1][0],path_arc=90*DEGREES), # A
            ShrinkToCenter(xtyt),
            Merge([entry[0][1] for entry in xtyt.get_entries()],atv[1][1]), # T
            ReplacementTransform(veq,atv[2]) # v
        ,run_time=1.75)
        self.wait(w)

        # merge components into full expression
        self.play(
            Merge([atv[1:],peqATv[3:]],peqATv[3:]),
            FadeOut(atv[:1],comps)
        ,run_time=1.75)
        self.wait(w)

        # expand to include outer products
        peqATvouter = MathTex(r"\mathbf{p}","=","A","A^T",r"\mathbf{v}","=",r"(\mathbf{x} \mathbf{x}^T + \mathbf{y} \mathbf{y}^T)",r"\mathbf{v}", font_size=80).move_to(peqATv)
        AlignBaseline(peqATvouter,peqATv)
        color_tex_standard(peqATvouter)
        self.play(ReplacementTransform(peqATv[:],peqATvouter[:5]))
        self.play(Write(peqATvouter[5]))
        self.play(FadeIn(peqATvouter[6:]))
        self.wait(w)

        # to projection matrices
        outers = MathTex("A","A^T","=",r"\mathbf{x} \mathbf{x}^T + \mathbf{y} \mathbf{y}^T",font_size=80).move_to(peqATvouter)
        color_tex_standard(outers)
        AlignBaseline(outers,peqATvouter)
        self.play(
            FadeOut(peqATvouter[0:2],peqATvouter[4],peqATvouter[6][0],peqATvouter[6][-1],peqATvouter[7],shift=DOWN), # p=,v,(),v
            ReplacementTransform(peqATvouter[2:4],outers[0:2]), # aat
            ReplacementTransform(peqATvouter[5],outers[2]), # =
            ReplacementTransform(peqATvouter[6][1:-1],outers[3][:]) # xxt+yyt
        ,run_time=1.5)
        self.wait(w)

        # to black
        self.play(FadeOut(*self.mobjects))


        


class Project2dOrthogonal(MovingCameraScene):
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
        ra = VGroup(
            Line(axes @ (0.9*pcoords),axes @ (0.9*pcoords+OUT*0.1), stroke_width=2),
            Line(axes @ (0.9*pcoords+OUT*0.1),axes @ (1*pcoords+OUT*0.1), stroke_width=2)
        )
        vectors = VGroup(v,x,y,p,px,py,r)
        dashes = VGroup(dy,dx)

        plane = Surface(lambda u,v:axes @ (u,v,0),u_range=[-0.25,1.25],v_range=[-0.25,1.25],resolution=1).set_opacity(0.5)
        
        diagram = VGroup(axes,plane,vectors,dashes, ra)
        diagram.rotate(-125*DEGREES).rotate(-70*DEGREES,RIGHT)        
        
        vl = MathTex(r"\mathbf{v}", color=VCOLOR, font_size=50).next_to(v.get_end(),buff=0.15)
        xl = MathTex(r"\mathbf{x}", color=XCOLOR, font_size=50).next_to(x.get_end(),LEFT,buff=0.15)
        yl = MathTex(r"\mathbf{y}", color=YCOLOR, font_size=50).next_to(y.get_end(),RIGHT,buff=0.15)
        pl = MathTex(r"\mathbf{p}", color=PCOLOR, font_size=50).next_to(p.get_end(),DR,buff=0.15)
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),UP,buff=0.25)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UR,buff=0.15)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl)
        diagram.add(labels)        
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.5+RIGHT*0.2)
        
        frame = self.camera.frame
        frame.save_state()
        frame.scale(0.48)
        for vector in (vector for vector in vectors): face_camera(self,vector)
        

        # write case 2, then move down
        c2 = MathTex(r"\text{Case 2: }", r"\mathbf{x}, \mathbf{y} \text{ Orthogonal}")
        color_tex_standard(c2)  
        diagram.shift(VGroup(diagram.copy(),c2.copy()).arrange(DOWN)[0].get_center() - diagram.get_center())      
        self.play(Write(c2[0]))
        self.play(Write(c2[1]))
        self.wait(w)
        
        # move text down
        self.play(c2.animate.next_to(diagram,DOWN))
        self.play(FadeIn(diagram))        
        self.wait(w)

        # zoom to x, then lengthen x
        self.play(frame.animate.move_to(xl).scale(0.3),run_time=1.5)
        self.play(
            x.animate.scale(1.2,about_point=x.points[0]),
            xl.animate.shift(0.3*(x.points[-1]-x.points[0])/np.linalg.norm(x.points[-1]-x.points[0]))
        )

        # zoom to y then lengthen y
        self.play(frame.animate.move_to(y.points[-1]).scale(0.8),run_time=2)
        self.play(
            y.animate.scale(0.85,about_point=y.points[0]),
            yl.animate.shift(-0.2*(y.points[-1]-y.points[0])/np.linalg.norm(y.points[-1]-y.points[0]))
        )
        self.wait(w)

        # zoom back out
        self.play(
            Restore(frame),
            VGroup(diagram,c2).animate.to_corner(UR),
            run_time=2.5
        )
        self.wait(w)

        # add components
        nex = MathTex(r"p_x","=", r"\frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}}", font_size=65).shift(3*LEFT+UP*1.5)  
        color_tex_standard(nex)
        ney = MathTex(r"p_y","=", r"\frac{\mathbf{y} \cdot \mathbf{v}}{\mathbf{y} \cdot \mathbf{y}}", font_size=65).shift(3*LEFT+DOWN*1.5)        
        color_tex_standard(ney)
        self.play(FadeIn(nex,ney,shift=LEFT))
        self.wait(w)

        # show full equation
        pe1 = MathTex(r"\mathbf{p}","=",r"p_x \mathbf{x}","+",r"p_y \mathbf{y}", font_size=60).next_to(c2,DOWN)
        color_tex_standard(pe1)  
        self.play(FadeIn(pe1))
        self.wait(w)
        
        # substitute components into formula
        pe2 = MathTex(r"\mathbf{p}","=",r"\frac{\mathbf{x} \cdot \mathbf{v}}{\mathbf{x} \cdot \mathbf{x}} \mathbf{x}","+",r"\frac{\mathbf{y} \cdot \mathbf{v}}{\mathbf{y} \cdot \mathbf{y}} \mathbf{y}", font_size=60).next_to(c2,DOWN)
        color_tex_standard(pe2)  
        self.play(
            *TransformBuilder(pe1,pe2,[(0,0),(1,1),(3,3),([2,-1],[2,-1]),([4,-1],[4,-1])]),
            FadeOut(pe1[2][:-1],pe1[4][:-1], shift=DOWN),
            *[TransformFromCopy(mob1,mob2,path_arc=-190*DEGREES) for mob1, mob2 in zip(nex[2],pe2[2][:-1])],
            *[TransformFromCopy(mob1,mob2,path_arc=-120*DEGREES) for mob1, mob2 in zip(ney[2],pe2[4][:-1])]
        ,run_time=2.5)
        self.wait(w)

        # to matrix
        nex2 = MathTex(r"p_x","=", r"\frac{\mathbf{x}^T \mathbf{v}}{\mathbf{x}^T \mathbf{x}}", font_size=65).move_to(nex,aligned_edge=LEFT)
        color_tex_standard(nex2)
        ney2 = MathTex(r"p_y","=", r"\frac{\mathbf{y}^T \mathbf{v}}{\mathbf{y}^T \mathbf{y}}", font_size=65).move_to(ney,aligned_edge=LEFT)
        color_tex_standard(ney2)
        pe3 = MathTex(r"\mathbf{p}","=",r"\frac{\mathbf{x}^T \mathbf{v}}{\mathbf{x}^T \mathbf{x}} \mathbf{x}","+",r"\frac{\mathbf{y}^T \mathbf{v}}{\mathbf{y}^T \mathbf{y}} \mathbf{y}", font_size=60).next_to(c2,DOWN)
        color_tex_standard(pe3)  
        self.play(
            ReplacementTransform(nex,nex2),
            ReplacementTransform(ney,ney2),
            ReplacementTransform(pe2,pe3)
        )
        self.wait(w)

        # move componentss up
        self.play(
            nex2.animate.shift(UP),
            ney2.animate.next_to(nex2.copy().shift(UP),DOWN)
        )

        # make p bigger
        pe4 = MathTex(r"\mathbf{p}","=",r"\frac{\mathbf{x}^T \mathbf{v}}{\mathbf{x}^T \mathbf{x}} \mathbf{x}","+",r"\frac{\mathbf{y}^T \mathbf{v}}{\mathbf{y}^T \mathbf{y}} \mathbf{y}", font_size=60)
        color_tex_standard(pe4).scale(80/60).next_to(c2.copy().shift(LEFT*c2.get_center()[0]), DOWN)
        self.play(
            ReplacementTransform(pe3, pe4),
            FadeOut(nex2,ney2)
        ,run_time=1.75)
        self.wait(w)

        # re-arrange
        pe5 = MathTex(r"\mathbf{p}","=",r"\frac{1}{\mathbf{x}^T \mathbf{x}} \mathbf{x} \mathbf{x}^T \mathbf{v}","+",r"\frac{1}{\mathbf{y}^T \mathbf{y}} \mathbf{y} \mathbf{y}^T \mathbf{v}", font_size=80).move_to(pe4)
        color_tex_standard(pe5)
        AlignBaseline(pe5,pe4)
        self.play(*TransformBuilder(
            pe4,pe5,
            [
                (0,0),(1,1),(3,3), # p= +
                ([2,slice(None,3)],[2,slice(6,None)]), # xtv
                ([2,3],[2,1]), # fraction
                (None,[2,0],FadeIn,{"shift":LEFT}), # 1
                ([2,slice(4,7)],[2,slice(2,5)],None,{"path_arc":90*DEGREES}), # xtx
                ([2,7],[2,5],None,{"path_arc":-240*DEGREES}), # x
                ([4,slice(None,3)],[4,slice(6,None)]), # ytv
                ([4,3],[4,1]), # fraction
                (None,[4,0],FadeIn,{"shift":LEFT}), # 1
                ([4,slice(4,7)],[4,slice(2,5)],None,{"path_arc":90*DEGREES}), # yty
                ([4,7],[4,5],None,{"path_arc":-240*DEGREES}), # y
            ]
        ),run_time=2)
        self.wait(w)

        # factor out v
        pe6 = MathTex(r"\mathbf{p}","=",r"\left( \frac{1}{\mathbf{x}^T \mathbf{x}} \mathbf{x} \mathbf{x}^T","+",r"\frac{1}{\mathbf{y}^T \mathbf{y}} \mathbf{y} \mathbf{y}^T \right)",r"\mathbf{v}", font_size=80).move_to(pe5)
        color_tex_standard(pe6)
        AlignBaseline(pe6,pe5)
        self.play(*TransformBuilder(
            pe5,pe6,
            [
                (0,0),(1,1),(3,3), # p= +
                (None,[2,0]), (None,[4,-1]), # ()
                ([2,slice(None,-1)],[2,slice(1,None)]), # x expression
                ([4,slice(None,-1)],[4,slice(None,-1)]), # y expression
            ]
        ),  Merge([pe5[2][-1],pe5[4][-1],],pe6[-1][0],{"path_arc":120*DEGREES})
        , run_time=2)
        self.wait(w)

        # indicate fractions
        self.play(
            Indicate(pe6[2][1:6],scale_factor=1.5),
            Indicate(pe6[4][0:5],scale_factor=1.5)
        )
        self.wait(w)

        # collapse to projector
        pe7 = MathTex(r"\mathbf{p}","=",r"A \left( A^T A \right)^{-1} A^T",r"\mathbf{v}", font_size=80).move_to(pe6)
        color_tex_standard(pe7)
        AlignBaseline(pe7,pe6)
        self.play(*TransformBuilder(
            pe6,pe7,
            [
                (0,0),(1,1),(-1,-1), # p= v
                (slice(2,5),2), # projection matrix
            ]
        ), run_time=1.5)
        self.wait(w)

        # bring components back
        self.play(FadeIn(nex2,ney2))
        self.wait(w)

        # component matrix
        comps = Matrix([
            ["p_x"],
            ["p_y"]
        ],v_buff=1.1,element_to_mobject_config={"font_size":65}).next_to(VGroup(nex2[1],ney2[1]),LEFT).shift(LEFT)
        color_tex_standard(comps)
        self.play(
            FadeIn(comps.get_brackets()),
            ReplacementTransform(nex2[0],comps.get_entries()[0][0]), #x
            ReplacementTransform(ney2[0],comps.get_entries()[1][0]) #y
        )
        eq = MathTex("=",r"\left( A^T A \right)^{-1} A^T \mathbf{v}", font_size=65).next_to(comps,RIGHT)
        color_tex_standard(eq)
        self.play(Merge([nex2[1],ney2[1]],eq[0]))
        self.play(
            ReplacementTransform(VGroup(nex2[2],ney2[2]),eq[1]),
        )
        self.wait(w)

        # flash extra A
        self.play(Indicate(pe7[2][0], scale_factor=1.5))
        self.wait(w)

        # flash outer product
        self.play(
            Indicate(pe7[2][0],scale_factor=2,color=XKCD.BUBBLEGUM),
            Indicate(pe7[2][-2:],scale_factor=2,color=XKCD.BUBBLEGUM)
        ,run_time=1.75)
        self.wait(w)

        # flash inner product
        self.play(
            Indicate(pe7[2][1:-2],scale_factor=1.75,color=XKCD.LIGHTTURQUOISE)
        ,run_time=1.75)
        self.wait(w)

        # to black
        self.play(FadeOut(*self.mobjects))



class Project2dFull(MovingCameraScene):
    def construct(self):
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
        v = Arrow(axes @ ORIGIN, axes @ vcoords, buff=0, color=VCOLOR).set_stroke(width=6)        
        x = Arrow(axes @ ORIGIN, axes @ xcoords, buff=0, color=XCOLOR).set_stroke(width=6)
        xp = Arrow(axes @ ORIGIN, axes @ (1,0,0), buff=0, color=XCOLOR).set_stroke(width=6)
        y = Arrow(axes @ ORIGIN, axes @ ycoords, buff=0, color=YCOLOR).set_stroke(width=6)        
        yp = Arrow(axes @ ORIGIN, axes @ (0,1,0), buff=0, color=YCOLOR).set_stroke(width=6)
        p = Arrow(axes @ ORIGIN, axes @ pcoords, buff=0, color=PCOLOR).set_stroke(width=6)        
        px = Arrow(axes @ ORIGIN, axes @ (pxcoord*xcoords), buff=0,color=PCOLOR).set_stroke(width=6)
        py = Arrow(axes @ ORIGIN, axes @ (pycoord*ycoords), buff=0,color=PCOLOR).set_stroke(width=6)
        dy = DashedLine(axes @ pcoords, axes @ (pxcoord*xcoords), dash_length=0.15).set_opacity(0.4)
        dx = DashedLine(axes @ pcoords, axes @ (pycoord*ycoords), dash_length=0.15).set_opacity(0.4)
        r = Arrow(axes @ pcoords, axes @ vcoords, buff=0, color=RCOLOR).set_stroke(width=30)        
        ArrowGradient(r,[PCOLOR,VCOLOR])
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
        pxl = MathTex(r"p_x \mathbf{x}", font_size=40).next_to(px.get_end(),LEFT,buff=0.15)
        color_tex_standard(pxl)        
        pyl = MathTex(r"p_y \mathbf{y}", font_size=40).next_to(py.get_end(),UR,buff=0.15)
        color_tex_standard(pyl)        
        rl = MathTex(r"\mathbf{v-p}", font_size=50).next_to(r,RIGHT,buff=0.15).shift(UP*0.3)
        color_tex_standard(rl)
        labels = VGroup(xl,vl,yl,pl,pxl,pyl,rl,xpl,ypl)
        diagram.add(labels)        
        
        diagram.shift(-VGroup(v,p,r).get_center()).shift(UP*0.5+RIGHT*0.2)
        
        frame = self.camera.frame
        frame.save_state()
        frame.scale(0.4)
        for vector in (vector for vector in vectors): face_camera(self,vector)
        
        # add v        
        self.play(FadeIn(v))
        self.play(Write(vl))
        self.play(Create(plane),run_time=2)
        self.wait(w)        

        # add xp, yp
        self.play(
            GrowArrow(xp),
            GrowArrow(yp),
            Write(xpl),
            Write(ypl),
            run_time=1.5
        )
        self.wait(w)

        # to not orthogonal
        self.play(
            ReplacementTransform(xp,x),
            ReplacementTransform(yp,y),
            ReplacementTransform(xpl,xl),
            ReplacementTransform(ypl,yl)
        ,run_time=1.75)
        self.wait(w)

        # case 3
        c3 = MathTex(r"\text{Case 3: }", r"\text{Any basis } \mathbf{x}, \mathbf{y}").next_to(diagram,DOWN)
        color_tex_standard(c3)
        self.play(Write(c3[0]))
        self.play(Write(c3[1]))
        self.wait(w)

        # add p
        self.play(TransformFromCopy(v,p),run_time=1.5)
        self.play(Write(pl))
        self.wait(w)

        # to px, py
        self.play(
            TransformFromCopy(p,px),
            Create(dy)
        , run_time=1.5)
        self.play(Write(pxl))
        self.play(
            TransformFromCopy(p,py),
            Create(dx)
        , run_time=1.5)
        self.play(Write(pyl))
        self.wait(w)

        # zoom in
        self.play(
            frame.animate.scale(0.3).move_to([4.2,1.6,0.5]).set_euler_angles(phi=-40*DEGREES, theta=-27.4*DEGREES, gamma=-25*DEGREES)
        ,run_time=4)
        self.wait(w)

        # zoom back out        
        self.play(frame.animate.scale(1/0.3).set_euler_angles(phi=0,theta=0,gamma=0).move_to(diagram).shift(DOWN*0.35),run_time=3)
        self.wait(w)

        # TODO: MovingCameraScene doesn't render 3d right, but 3dscene leaves stroke width unchanged when zooming in, making stuff look weird,
        # so, change this (and previous scenes :/) over to threedscene, but adjust arrow strokes to compensate?
        # ...or does that create a problem for when we zoom away from the item?




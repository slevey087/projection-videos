from manim import *
from manim.opengl import *
from scipy import stats
import numpy as np


from fix_opengl_vector import *


def window(arr, k):
    """Utility function to window over a list. arr = list to window, k = length of window.    """
    for i in range(len(arr)-k+1):
        yield arr[i:i+k]


def color_tex(tex, *color_pairs, tolerance=None):
    """
        Color Manim Tex expressions by symbol. 
        tex = the Tex item you want to color (MathTex, Tex, or Matrix)
        *color_pairs = tuples of the form `(symbol, color)`, where `symbol` is a string and `color` is the Manim color to color it
    """
    # recurse if VGroup
    if isinstance(tex, VGroup):
        for item in tex: color_tex(item, *color_pairs)
        return tex

    # check for single color pair
    if len(color_pairs) == 2 and isinstance(color_pairs[1], ManimColor):
        color_pairs = [(color_pairs[0], color_pairs[1])]    

    # do different things depending on type of input
    if isinstance(tex, MathTex):
        for subtex in tex:
            color_tex(subtex, *color_pairs)        
    elif isinstance(tex, SingleStringMathTex):        
        for string, color in color_pairs:
            string = SingleStringMathTex(string, tex_environment=tex.tex_environment, font_size=tex.font_size)
            length = len(string)

            for win in window(tex, length):            
                if np.all([np.allclose(char.move_to(stroke).points, stroke.points,atol=0.05) if (char.points.shape == stroke.points.shape) else False for char, stroke in zip(string, win)]):                
                    win.set_color(color)
    elif isinstance(tex, Matrix):
        for entry in tex.get_entries():
            color_tex(entry, *color_pairs)
    
    return tex


def TransformBuilder(mobject1, mobject2, formula, default_transform=ReplacementTransform, default_creator=Write, default_remover=FadeOut):
    """
        Returns transformations which convert mobject1 into mobject2, according to formula.

        formula should be a list of tuples, one for each transformation.
        A tuple consists of four elements: a locator for the initial mobject portion, a locator for the final mobject portion, and a code for what kind of transform to do, and any optional kwargs for the transform as a dict
        To write a new element, put None for the source.
        To remove an existing element, put None for the destination.

        A formula is a list of numbers, where each number is a deeper resource locator for a mobject. The last entry can be a slice() or list to select groups.
    """
    def recurse(l, locator):
        if len(locator) == 1: 
            if isinstance(locator[0], list): return VGroup(*[l[i] for i in locator[0]])
            else: return l[locator[0]]
        else: return recurse(l[locator[0]], locator[1:])

    anims = []
    for item in formula:
        source_locator      = item[0]
        destination_locator = item[1]
        transform           = item[2] if len(item) >= 3 else None
        kwargs              = item[3] if len(item) == 4 else {}
        
        # Figure out source
        if source_locator is None: source = None
        elif not isinstance(source_locator, list): source = mobject1[source_locator]
        else: source = recurse(mobject1, source_locator)

        # Figure out destination
        if destination_locator is None: destination = None
        elif not isinstance(destination_locator, list): destination = mobject2[destination_locator]
        else: destination = recurse(mobject2, destination_locator)

        
        # Default transforms
        if source is None:
            transform = transform or default_creator
            anims.append(transform(destination, **kwargs))
        
        elif destination is None:
            transform = transform or default_remover
            anims.append(transform(source, **kwargs))
        
        else:
            transform = transform or default_transform
            anims.append(transform(source, destination, **kwargs))
    
    return anims


class Merge(AnimationGroup):
    """
        Animation to merge a group of mobjects into one.

        Although each mobject is transformed into the target, 
        at the end of the animation there will only be one copy of the target.

        Parameters
        ----------
        group
            List of mobjects to be merged
        target_mobject
            Mobject that should be left at the end
        animargs
            Arguments to be passed to underlying transforms
        kwargs
            Arguments to be passed to AnimationGroup
    """
    def __init__(self, group, target_mobject, animargs={}, **kwargs):        
        self.primary_mobject = group[0]
        self.primary_target = target_mobject
        self.secondary_mobjects = group[1:]
        self.secondary_targets = [target_mobject.copy() for mobject in group[1:]]        
        
        transforms = [ReplacementTransform(self.primary_mobject,self.primary_target, **animargs)]
        transforms += [Transform(mob, target, remover=True,**animargs) for mob, target in zip(self.secondary_mobjects, self.secondary_targets)]

        super().__init__(
            *transforms, **kwargs    
        )



def GetBaseline(item):
    """
        Finds the vertical position of Tex according to character.
        (Works by creating a phantom copy with a plus sign, then finding its bottom).
        Returns single number, height of baseline.
    """
    orig_text = item.get_tex_string()
    
    # helps with bulleted lists
    orig_text = orig_text[:-2] if orig_text[-2:] == r"\\" else orig_text
    
    orig_center = item.get_center()
    orig_bottom = item.get_bottom()

    orig_env = item.tex_environment
    
    if isinstance(item, Tex): temp_text = Tex(orig_text,"+", tex_environment=orig_env)
    else: temp_text = MathTex(orig_text,"+", tex_environment=orig_env)
    temp_text.scale((orig_center[1]-orig_bottom[1])/(temp_text[0].get_center()[1]-temp_text[0].get_bottom()[1]))
    temp_text.shift(orig_center-temp_text[0].get_center())
        
    baseline = temp_text[1].get_bottom()[1]
    return baseline



def AlignBaseline(item, reference):
    """
        Aligns two pieces of Tex by character baseline. First arguments gets moved vertically to baseline of 2nd argument.
    """
    BaselineDifference = GetBaseline(reference) - GetBaseline(item) 
    item.shift([0,BaselineDifference,0])
    return item


def FixMatrixBaselines(matrix,anchors):
    for row, anchor in zip(matrix.get_rows(), anchors):
        anchor_base = GetBaseline(row[anchor])
        for item in row:
            base = GetBaseline(item)
            item.shift([0,anchor_base-base,0])
    return matrix


def arrange_baselines(bulleted_list):
    """For use with BulletedList. Will evenly space items vertically according to text baseline."""
    baselines = [GetBaseline(item) for item in bulleted_list]
    top = baselines[0]
    bottom = baselines[-1]
    span = top - bottom
    gap = span / (len(baselines) - 1)
    for i, item in enumerate(bulleted_list[1:-1], 1): 
        new_base = top - gap * i
        item.shift([0,new_base - baselines[i],0])
    return bulleted_list


def compare_index_labels(scene, mob1, mob2, scale_factor=1.5):    
    """
        Displays index labels for two mobjects next to each other. For use when bulding complex animations.
    """
    
    mob1.scale(scale_factor), mob2.scale(scale_factor)
    VGroup(mob1, mob2).arrange(DOWN)
    scene.add(mob1, index_labels(mob1))
    scene.add(mob2, index_labels(mob2))




def matrix_product(matrix1, matrix2, h_buff):
    """
        Returns a matrix product. Doesn't actualy multiiply numbers, just writes entries next to each other.
    """
    rows1 = len(matrix1.get_rows())
    cols1 = len(matrix1.get_rows()[0])
    rows2 = len(matrix2.get_rows())
    cols2 = len(matrix2.get_rows()[0])
    if cols1 != rows2:
        raise ValueError(
            "The number of columns in matrix1 must be equal to the number of rows in matrix2.")
    entries1 = matrix1.get_entries()
    entries2 = matrix2.get_entries()
    entries3 = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            entry = ""
            for k in range(cols1):
                entry += entries1[i*cols1+k].get_tex_string() + \
                    entries2[k*cols2+j].get_tex_string()+"+"
            row.append(entry[:-1])
        entries3.append(row)
    return Matrix(entries3, h_buff=h_buff)



def multiply_matrix(scene, matrix1, matrix2, matrix3=None, h_buff=2.15, location=None):
    """
        Animates the product of two matrices, already on the screen.
        Returns (matrix3, equals_sign)
    """
    equals_sign = MathTex("=").next_to(
        matrix2 if location is None else location, RIGHT)
    matrix3 = matrix_product(
        matrix1, matrix2, h_buff=h_buff) if matrix3 is None else matrix3
    matrix3 = matrix3.next_to(equals_sign, RIGHT)

    scene.play(Write(equals_sign), Write(matrix3.get_brackets()))

    for i in range(len(matrix1.get_rows())):
        for j in range(len(matrix2.get_rows()[0])):
            row = matrix1.get_rows()[i]
            col = matrix2.get_columns()[j]
            source = VGroup(row, col)
            scene.play(Indicate(row), Indicate(col), TransformFromCopy(
                source, matrix3.get_rows()[i][j]))

    return (matrix3, equals_sign)


def transpose3(scene, matrix, with_diagonal=False, diagonal_flash=False, fade_when_done=False):
    # Color blue, transpose, color white
    if with_diagonal:
        diagonal = DashedLine(matrix.get_entries()[0].get_center(), matrix.get_entries()[
                              8].get_center(), dash_length=0.25, dashed_ratio=0.6)
        scene.play(Create(diagonal))

        if diagonal_flash:
            scene.remove(diagonal)
            scene.play(Create(diagonal))

    # Turn blue
    elements = matrix.get_entries()
    off_elements = [elements[i] for i in [1, 2, 3, 5, 6, 7]]
    scene.play(*[element.animate.set_color(BLUE)
               for element in off_elements], run_time=0.5)

    # Move
    tmap = {"origin": [1, 2, 3, 5, 6, 7], "destination": [3, 6, 1, 7, 2, 5]}
    entry_animations = [elements[origin].animate.move_to(elements[destination].get_center(
    )) for (origin, destination) in zip(tmap["origin"], tmap["destination"])]
    scene.play(*entry_animations, run_time=1.3)

    # Turn white
    scene.play(*[element.animate.set_color(WHITE)
               for element in off_elements], run_time=0.5)

    fades = []
    if with_diagonal:
        fades += [FadeOut(diagonal)]
    if fade_when_done:
        fades += [FadeOut(matrix)]
    if len(fades):
        scene.play(AnimationGroup(*fades))
    return matrix


def transpose4(scene, matrix, other_anims=[]):
    # Color blue, transpose, color white
    
    # Turn blue
    elements = matrix.get_entries()
    off_elements = [elements[i] for i in [1, 2, 3,4, 6, 7,8,9,11,12,13,14]]
    scene.play(*[element.animate.set_color(BLUE)
               for element in off_elements], run_time=0.5)

    # Move
    tmap = {"origin": [1, 2, 3,4, 6, 7,8,9,11,12,13,14], "destination": [4,8, 12, 1, 9, 13, 2, 6, 14, 3, 7, 11]}
    entry_animations = [elements[origin].animate.move_to(elements[destination].get_center(
    )) for (origin, destination) in zip(tmap["origin"], tmap["destination"])]
    scene.play(*entry_animations,*other_anims, run_time=1.3)

    # Turn white
    scene.play(*[element.animate.set_color(WHITE)
               for element in off_elements], run_time=0.5)
    
    
    return matrix


def project(vector, projector):
    """
        Projects 1st vector along direction of 2nd. Both vectors are lists of numbers. Any dimension.
    """
    return np.array(projector) * (np.dot(vector, projector) / np.dot(projector, projector))



def fill_space_with_vectors(axes, x_dims=(-6,6,1), y_dims=(-3,4,1), z_dims=(-3,4,1), color=BLUE_B):
    import itertools
    if isinstance(axes, ThreeDAxes):
        vectors = [[[
            Arrow(axes.c2p(0,0,0), axes.c2p(i,j,k), buff=0, color=color)
                for k in np.arange(*z_dims)] for j in np.arange(*y_dims)] for i in np.arange(*x_dims)
        ]
        anims = [GrowArrow(vector) for vector in itertools.chain(*itertools.chain(*vectors))]
    else:
        vectors = [[
            Arrow(axes.c2p(0,0), axes.c2p(i,j), buff=0, color=color)
                for j in np.arange(*y_dims)] for i in np.arange(*x_dims)
        ]
        anims = [GrowArrow(vector) for vector in itertools.chain(*vectors)]
    
    return vectors, anims
    

def project_vector_space(axes, original_vectors, along_vector1, along_vector2=None, color=BLUE_E):
    import itertools  
    if isinstance(axes, ThreeDAxes):        
        if along_vector2 is not None:
            projected_vectors = [[[Arrow(axes.c2p(0,0), axes.c2p(*(project(axes.p2c(vector.get_end()), along_vector1)+project(axes.p2c(vector.get_end()), along_vector2))), buff=0, color=color) for vector in col] for col in row] for row in original_vectors]
        else:
            projected_vectors = [[[Arrow(axes.c2p(0,0), axes.c2p(*(project(axes.p2c(vector.get_end()), along_vector1))), buff=0, color=color) for vector in col] for col in row] for row in original_vectors]
        anims = [ReplacementTransform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*itertools.chain(*original_vectors)), itertools.chain(*itertools.chain(*projected_vectors)))]
    else: 
        projected_vectors = [[Arrow(axes.c2p(0,0), axes.c2p(*project(axes.p2c(vector.get_end()), along_vector1)), buff=0, color=color) for vector in row] for row in original_vectors]
        anims = [ReplacementTransform(vector, projected_vector) for vector, projected_vector in zip(itertools.chain(*original_vectors), itertools.chain(*projected_vectors))]
    
    return projected_vectors, anims
    


class ParallelAlongPath(Animation):
    def __init__(self, mobject, path, **kwargs):
        super().__init__(mobject, **kwargs)
        self.path = path
        self.initial_mobject_angle = self._get_mobject_angle()        
        self.initial_path_angle = self._get_path_angle(0)        
        self.about = self.mobject
    

    def _get_mobject_angle(self):        
        return angle_of_vector(self.mobject.family_members_with_points()[0].points[3]-self.mobject.family_members_with_points()[0].points[0])

    def _get_path_angle(self, alpha):
        t0 = self.path.t_min + self.rate_func(alpha) * (self.path.t_max - self.path.t_min)
        t1 = t0 + self.path.t_step
        # p0 = self.path.get_point_from_function(t0)
        # p1 = self.path.get_point_from_function(t1)
        p0 = self.path.point_from_proportion(self.rate_func(alpha))
        p1 = self.path.point_from_proportion(self.rate_func(alpha+0.00001)) if alpha < 1 else 2*self.path.point_from_proportion(self.rate_func(alpha)) - self.path.point_from_proportion(self.rate_func(alpha-0.00001))
        tangent = p1-p0
        return angle_of_vector(tangent)


    def interpolate_mobject(self, alpha: float) -> None:
        # rotate
        change = self._get_path_angle(alpha)-self.initial_path_angle                
        cum_rotation = self._get_mobject_angle() - self.initial_mobject_angle

        self.mobject.rotate(change - cum_rotation,about_point=self.about.get_center())
        
        # move
        point = self.path.point_from_proportion(self.rate_func(alpha))
        self.mobject.move_to(point)


class FollowRotation(Animation):
    def __init__(self, mobject, path,about_mob=None, reverse_angle=True,subtract_180=False,**kwargs):
        super().__init__(mobject, **kwargs)
        self.path = path        
        self.reverse_angle=reverse_angle,
        self.subtract_180 = subtract_180,
        self.about = about_mob or self.mobject            
    

    def _get_path_angle(self, alpha):
        # t0 = self.path.t_min + self.rate_func(alpha) * (self.path.t_max - self.path.t_min)
        # t1 = t0 + self.path.t_step
        # p0 = self.path.get_point_from_function(t0)
        # p1 = self.path.get_point_from_function(t1)
        p0 = self.path.point_from_proportion(self.rate_func(alpha))
        p1 = self.path.point_from_proportion(self.rate_func(alpha+0.00001)) if alpha < 1 else 2*self.path.point_from_proportion(self.rate_func(alpha)) - self.path.point_from_proportion(self.rate_func(alpha-0.00001))
        tangent = p1-p0
        return angle_of_vector(tangent)

    def interpolate_mobject(self, alpha: float) -> None:
        current_path_angle = self._get_path_angle(alpha)
        
        change = current_path_angle - (180*DEGREES if self.subtract_180 else 0)        
        if self.reverse_angle: change = -change
        
        self.mobject.rotate(change,about_point=self.about.get_center())


def Dice(num=None):   
    # adopted from https://stackoverflow.com/questions/76516060/manim-dice-simulation
    faces = VGroup(
        VGroup(
            Square(side_length=2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([0,0,0], radius=0.2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([-0.67,-0.67,0], radius=0.2),
            Dot([+0.67,+0.67,0], radius=0.2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([-0.67,-0.67,0], radius=0.2),
            Dot([0,0,0], radius=0.2),
            Dot([+0.67,+0.67,0], radius=0.2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([-0.67,-0.67,0], radius=0.2),
            Dot([+0.67,+0.67,0], radius=0.2),
            Dot([-0.67,+0.67,0], radius=0.2),
            Dot([+0.67,-0.67,0], radius=0.2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([-0.67,-0.67,0], radius=0.2),
            Dot([+0.67,+0.67,0], radius=0.2),
            Dot([0,0,0], radius=0.2),
            Dot([-0.67,+0.67,0], radius=0.2),
            Dot([+0.67,-0.67,0], radius=0.2),
        ),
        VGroup(
            Square(side_length=2),
            Dot([-0.67,-0.67,0], radius=0.2),
            Dot([+0.67,+0.67,0], radius=0.2),
            Dot([-0.67,+0.67,0], radius=0.2),
            Dot([+0.67,-0.67,0], radius=0.2),
            Dot([+0.67,0,0], radius=0.2),
            Dot([-0.67,0,0], radius=0.2),
        ),
    )    
    return faces if num is None else faces[num]        




def box_to_3D_point(point):
    """
        Creates dashed lines for each edge in a box connecting the origin to a 3d point - for making 3d points more obvious
    """
    x_bottom = [point[0],0,0]
    y_bottom = [0,point[1],0]
    xy_bottom = [point[0],point[1],0]
    x_top = [point[0],0,point[2]]
    y_top = [0,point[1],point[2]]
    z_top = [0,0,point[2]]
    return VGroup(
        DashedLine(ORIGIN,x_bottom),
        DashedLine(ORIGIN,y_bottom),
        DashedLine(x_bottom,xy_bottom),
        DashedLine(y_bottom,xy_bottom),
        DashedLine(x_bottom,x_top),
        DashedLine(y_bottom,y_top),
        DashedLine(x_top,point),
        DashedLine(y_top,point),
        DashedLine(z_top,x_top),
        DashedLine(z_top,y_top),
        DashedLine(xy_bottom,point)
)


# from fix_opengl_vector import *
def NumberPlane_to_3D(numberplane, color=None, opacity=0.5):
    """ 
        Hack to make depth work. Just returns OpenGL group of 3d lines where lines should be
    """
    return OpenGLGroup(
            *[Line3D(*line.get_start_and_end(), thickness=0.01, color=color).set_opacity(opacity) for line in numberplane[1]]
        )        


def Axes_to_3D(axes):
    """
        Similar hack for OpenGL axes that make depth work.
    """
    new_axes = OpenGLGroup(
        *[Arrow3D(*axis.get_start_and_end(), thickness=0.01) for axis in axes]
    )

    for axis, new_axis in zip(axes, new_axes):
        ticks = OpenGLGroup(
            *[Line3D(*tick.get_start_and_end()) for tick in axis.ticks]
        )
        new_axis.add(ticks)
        new_axis.ticks = ticks

    new_axes.c2p = axes.c2p
    new_axes.p2c = axes.p2c
    new_axes.get_origin = axes.get_origin

    return new_axes


class DashedLine3D(OpenGLGroup):
    def __init__(self, start, end, dash_length=0.05, dashed_ratio=0.5, line_kwargs={}, **kwargs):
        cycle_length = dash_length / dashed_ratio        
        displacement_vec = np.asarray(end) - np.asarray(start)
        displacement_hat = displacement_vec/np.linalg.norm(displacement_vec)
        num_cycles = np.linalg.norm(displacement_vec) / cycle_length                
        complete_cycles = int(np.floor(num_cycles))
        

        super().__init__(*[], **kwargs)
        for i in range(complete_cycles):
            self.add(Line3D(start+i*cycle_length*displacement_hat, start+(dashed_ratio+i)*cycle_length*displacement_hat, **line_kwargs))                        
        
        if num_cycles != complete_cycles:
            i = complete_cycles
            if (num_cycles - complete_cycles) > dashed_ratio:
                self.add(Line3D(start+i*cycle_length*displacement_hat, start+(i+dashed_ratio)*cycle_length*displacement_hat, **line_kwargs))

            else:                
                self.add(Line3D(start+i*cycle_length*displacement_hat, end, **line_kwargs))



class RightAngle3D(OpenGLGroup):
    def __init__(self, line1, line2, length = 0.2, threeD=True, **kwargs):
        # Assumes that line 2 starts where line 1 ends
        
        line1_start = line1.get_start()
        line1_end = line1.get_end()
        line2_start = line2.get_start()
        line2_end = line2.get_end()

        unit1 = (line1_end-line1_start)/np.linalg.norm(line1_end-line1_start)
        start = line1_end - unit1 * length
        unit2 = (line2_end - line2_start)/np.linalg.norm(line2_end - line2_start)
        end = line2_start + unit2 * length

        direct = end - start
        to_intersection = line1_end-start
        
        middle = 2*project(to_intersection,direct)-to_intersection + start                

        if threeD:
            parts = [Line3D(start,middle, **kwargs), Line3D(middle,end, **kwargs)]
        else:
            parts = [Line(start,middle, **kwargs).set_flat_stroke(False), Line(middle,end, **kwargs).set_flat_stroke(False)]                        
        super().__init__(*parts)


class RightAngleIn3D(OpenGLVMobject):
    def __init__(self, line1, line2, length = 0.2, **kwargs):
        # Assumes that line 2 starts where line 1 ends
        
        line1_start = line1.get_start()
        line1_end = line1.get_end()
        line2_start = line2.get_start()
        line2_end = line2.get_end()

        unit1 = (line1_end-line1_start)/np.linalg.norm(line1_end-line1_start)
        start = line1_end - unit1 * length
        unit2 = (line2_end - line2_start)/np.linalg.norm(line2_end - line2_start)
        end = line2_start + unit2 * length

        direct = end - start
        to_intersection = line1_end-start
        
        middle = 2*project(to_intersection,direct)-to_intersection + start                
        middle1 = start + (middle-start)/2
        middle2 = middle + (end - middle)/2
        
        super().__init__(**kwargs)
        self.points=np.array([start,middle1,middle,middle, middle2,end])

from manim import *
import numpy as np

class DerivativeVisualization(Scene):
    def construct(self):
        # PART 1: Setup axes and function
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 2, 1],
            x_length=5.5,
            y_length=4.5,
            axis_config={"color": GRAY}
        ).shift(LEFT * 2.5)
        
        # Define the function
        def f(x):
            return x**3 * np.exp(-x)
        
        # Plot the function
        graph = axes.plot(f, color=BLUE, x_range=[-3.5, 3.5])
        func_label = MathTex(r"f(x) = x^{3}e^{-x}", color=BLUE, font_size=36).to_corner(UL).shift(DOWN*0.3)
        
        # PART 2: CALCULATION STEPS (Right side, no overlap)
        calc_title = Text("Derivative Calculation:", font_size=20, color=WHITE).to_edge(RIGHT).shift(LEFT*0.2 + UP*3.3)
        
        calc_step1 = MathTex(r"f(x) = x^{3}e^{-x}", font_size=22).next_to(calc_title, DOWN, buff=0.3, aligned_edge=LEFT)
        calc_step2 = MathTex(r"\text{Product Rule: } (uv)' = u'v + uv'", font_size=22).next_to(calc_step1, DOWN, buff=0.25, aligned_edge=LEFT)
        calc_step3 = MathTex(r"f'(x) = 3x^{2}e^{-x} - x^{3}e^{-x}", font_size=22).next_to(calc_step2, DOWN, buff=0.25, aligned_edge=LEFT)
        calc_final = MathTex(r"f'(x) = x^{2}e^{-x}(3-x)", font_size=26, color=YELLOW).next_to(calc_step3, DOWN, buff=0.3, aligned_edge=LEFT)
        
        # Ensure all calculation steps fit on screen
        calc_steps = VGroup(calc_title, calc_step1, calc_step2, calc_step3, calc_final)
        if calc_steps.width > 3.8:
            calc_steps.scale_to_fit_width(3.8)
        
        # PART 3: Moving elements
        x_tracker = ValueTracker(-2)
        
        # Define derivative function
        def f_prime(x):
            return x**2 * np.exp(-x) * (3 - x)
        
        # Moving dot on curve
        dot = always_redraw(lambda: Dot(
            axes.c2p(x_tracker.get_value(), f(x_tracker.get_value())),
            color=RED,
            radius=0.08
        ))
        
        # Tangent line
        tangent = always_redraw(lambda: axes.plot(
            lambda x: f_prime(x_tracker.get_value()) * (x - x_tracker.get_value()) + f(x_tracker.get_value()),
            x_range=[x_tracker.get_value()-1.5, x_tracker.get_value()+1.5],
            color=GREEN
        ))
        
        # Derivative value display - positioned BELOW calculation to avoid overlap
        deriv_label = always_redraw(lambda: MathTex(
            r"f'({:.1f}) = {:.2f}".format(x_tracker.get_value(), f_prime(x_tracker.get_value())),
            font_size=30,
            color=YELLOW
        ).next_to(calc_steps, DOWN, buff=0.5))
        
        # PART 4: Animation sequence
        self.play(Create(axes), Write(func_label), run_time=1)
        self.play(Create(graph), run_time=1.5)
        self.wait(0.5)
        
        # Show calculation steps one by one
        self.play(Write(calc_title), run_time=0.5)
        self.play(Write(calc_step1), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_step2), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_step3), run_time=0.7)
        self.wait(0.3)
        self.play(Write(calc_final), run_time=0.8)
        self.wait(0.5)
        
        # Add moving elements
        self.play(Create(dot), Create(tangent), Write(deriv_label), run_time=1)
        
        # Animate movement
        self.play(x_tracker.animate.set_value(2), run_time=3, rate_func=smooth)
        self.wait(1)
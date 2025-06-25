# sketch_analogy_truly_smooth.py
from manim import *
import numpy as np

class SketchAnalogySmooth(Scene):
    def construct(self):
        # 1. 基础场景设置
        split_line = Line(LEFT * 7.5, RIGHT * 7.5)

        # 上半部分：光模拟
        light_source = Dot(point=UP * 2 + LEFT * 5, color=YELLOW, radius=0.2)
        light_label = Tex("Light").next_to(light_source, DOWN)
        
        detector_rect = Rectangle(width=0.1, height=0.6, fill_color=BLUE_D, fill_opacity=1, stroke_color=WHITE)
        detector_normal = Line(ORIGIN, RIGHT * 0.5, color=WHITE).move_to(detector_rect.get_right(), aligned_edge=LEFT)
        detector = VGroup(detector_rect, detector_normal).move_to(UP * 2 + RIGHT * 1).rotate(PI, about_point=UP * 2 + RIGHT * 1)
        
        intensity_label = Tex("$I_x$ = ", font_size=40)
        intensity_value = DecimalNumber(0, num_decimal_places=2, font_size=40)

        # 下半部分：引力模拟
        mass_source = Dot(point=DOWN * 2 + LEFT * 5, color=BLUE_E, radius=0.3)
        mass_label = Tex("M").next_to(mass_source, DOWN)
        
        particle = Dot(DOWN * 2 + RIGHT * 1, color=RED, radius=0.1)
        particle_label = Tex("x")

        total_force_vec = DashedLine(color=GREEN, stroke_opacity=0.5)
        detection_dir_arrow = Arrow(color=WHITE, buff=0, max_tip_length_to_length_ratio=0.2, stroke_width=3)
        force_component_vec = Vector(color=YELLOW)
        force_label = Tex("$F_x$ = ", font_size=40)
        force_value = DecimalNumber(0, num_decimal_places=2, font_size=40)

        # 将所有动态更新的元素分组
        updatable_elements = VGroup(
            intensity_label, intensity_value, particle_label, total_force_vec, 
            detection_dir_arrow, force_component_vec, force_label, force_value
        )

        # 2. 核心更新逻辑
        def update_dependencies(mobj):
            """这个updater只根据detector和particle的当前状态来更新其他所有元素"""
            # --- 光模拟更新 ---
            vec_to_light = light_source.get_center() - detector.get_center()
            distance = np.linalg.norm(vec_to_light)
            dist_sq = distance**2
            
            detector_dir = detector[1].get_vector() # 获取方向线的向量
            cos_theta = np.dot(normalize(vec_to_light), normalize(detector_dir))
            
            brightness = (18 / dist_sq) * cos_theta if cos_theta > 0 else 0
            intensity_value.set_value(brightness)
            intensity_label.next_to(detector, RIGHT, buff=0.5)
            intensity_value.next_to(intensity_label, RIGHT, buff=0.1)

            # --- 引力模拟更新 ---
            particle_label.next_to(particle, UP, buff=0.2)
            vec_to_mass = mass_source.get_center() - particle.get_center()
            total_force_vec.put_start_and_end_on(particle.get_center(), mass_source.get_center())
            
            force_mag_total = 18 / np.sum(vec_to_mass**2)
            total_force_normalized = normalize(vec_to_mass)
            
            detection_dir_norm = normalize(detector_dir)
            detection_dir_arrow.become(Arrow(particle.get_center(), particle.get_center() + detection_dir_norm * 1.5, color=WHITE, buff=0, max_tip_length_to_length_ratio=0.2, stroke_width=3))

            force_component_mag = np.dot(total_force_normalized, detection_dir_norm) * force_mag_total
            if force_component_mag < 0: force_component_mag = 0
            force_component_vec.become(Vector(detection_dir_norm * force_component_mag * 0.5, color=YELLOW).move_to(particle.get_center(), aligned_edge=ORIGIN))
            
            force_value.set_value(force_component_mag)
            force_label.next_to(force_component_vec, RIGHT, buff=0.2)
            force_value.next_to(force_label, RIGHT, buff=0.1)

        # 3. 带解说的动画演示流程
        # 添加updater，它将在所有动画期间自动运行
        updatable_elements.add_updater(update_dependencies)

        # 开场：创建所有元素
        all_elements = VGroup(split_line, light_source, light_label, mass_source, mass_label, detector, particle, updatable_elements)
        self.play(FadeIn(all_elements, shift=UP*0.5), run_time=1.5)

        # --- 情况一：正对，基准距离 ---
        caption1 = Text("情况一：探测器正对源头，距离为R", font_size=32).to_edge(DOWN)
        self.play(Write(caption1))
        self.wait(2.5)

        # --- 情况二：正对，距离变远 ---
        caption2 = Text("情况二：保持方向不变，将距离增加到4R", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(caption1, caption2))
        
        # 使用 animate 方法来创建平滑移动
        self.play(
            detector.animate.move_to(UP * 2 + RIGHT * 4),
            particle.animate.move_to(DOWN * 2 + RIGHT * 4),
            run_time=4
        )
        self.wait(1)
        caption2_result = Text("强度和力按“平方反比”减小", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(caption2, caption2_result))
        self.wait(2.5)

        # --- 情况三：角度变化 ---
        caption3_return = Text("现在... 回到原始距离R", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(caption2_result, caption3_return))

        self.play(
            detector.animate.move_to(UP * 2 + RIGHT * 1),
            particle.animate.move_to(DOWN * 2 + RIGHT * 1),
            run_time=3
        )
        self.wait(1)

        caption3_rotate = Text("情况三：距离不变，将探测器旋转56°", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(caption3_return, caption3_rotate))

        # 使用 Rotate 动画类来实现平滑旋转
        self.play(
            Rotate(detector, angle=-(56 * DEGREES), about_point=detector.get_center()),
            run_time=4
        )
        self.wait(1)
        caption3_result = Text("强度和分力按“余弦定律”减小", font_size=32).to_edge(DOWN)
        self.play(ReplacementTransform(caption3_rotate, caption3_result))
        self.wait(3)

        # 清理工作
        updatable_elements.remove_updater(update_dependencies)
        self.play(FadeOut(all_elements, caption3_result))
        self.wait()

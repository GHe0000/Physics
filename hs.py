# holmberg_simulation_fixed.py
from manim import *
import numpy as np

# --- 模拟参数 ---
G = 800  # 引力常数 (调整以获得好的视觉效果)
TIME_STEP = 0.005  # 时间步长
N_BULBS_PER_GALAXY = 37 # 每个星系的灯泡数量
ROTATION_SPEED = 120 # 内部旋转速度
GALAXY_MASS = 100 # 星系总质量
GALAXY_RADIUS = 2.0 # 星系半径
INITIAL_VELOCITY = np.array([0., 0., 0.]) # 初始速度 (模拟抛物线轨迹)

class Bulb(Dot):
    """
    代表一个'质量元素'的灯泡类。
    继承自Manim的Dot，并增加了物理属性。
    """
    def __init__(self, position, mass, velocity, **kwargs):
        super().__init__(point=position, **kwargs)
        self.mass = mass
        self.velocity = np.array(velocity, dtype=float)
        self.force = np.zeros(3, dtype=float)

class HolmbergExperiment(Scene):
    def construct(self):
        # 1. 标题和引言
        self.show_title_and_intro()

        # 2. 建立星系
        galaxies = self.create_galaxies()
        galaxy_group = VGroup(*[bulb for galaxy in galaxies for bulb in galaxy])
        self.play(Create(galaxy_group), run_time=2)
        self.wait(1)

        # 3. 解释模拟原理
        self.explain_simulation_logic(galaxies)
        
        # 4. 运行主模拟
        self.run_simulation(galaxies, galaxy_group)

    def show_title_and_intro(self):
        """显示标题和介绍文本"""
        title = Text("Holmberg's 1941 Nebula Encounter Experiment").scale(0.8)
        subtitle = Text("Visualized with Python & Manim", t2c={'Python': BLUE, 'Manim': YELLOW}).scale(0.6).next_to(title, DOWN)
        
        intro_text = VGroup(
            Text("In 1941, Erik Holmberg simulated galaxy encounters."),
            Text("He used light bulbs to represent mass elements,"),
            Text("and a photocell to measure the 'gravitational' force.")
        ).scale(0.5).arrange(DOWN, aligned_edge=LEFT).to_corner(UL)

        self.play(Write(title), Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
        self.play(Write(intro_text))
        self.wait(3)
        self.play(FadeOut(intro_text))

    def create_galaxies(self):
        """创建两个星系（灯泡群）"""
        # 根据Holmberg论文中的图3b创建灯泡布局
        layout = [(0, 0)] # 中心
        for r in [0.8, 1.4, 2.0]: # 半径层
            n_in_ring = 6 if r < 1.0 else 12 if r < 1.8 else 18
            for i in range(n_in_ring):
                angle = 2 * PI * i / n_in_ring
                layout.append((r * np.cos(angle), r * np.sin(angle)))
        
        galaxy1_bulbs = []
        galaxy2_bulbs = []
        
        center1 = np.array([-4, 2, 0])
        center2 = np.array([4, -2, 0])

        for pos_x, pos_y in layout:
            dist = np.sqrt(pos_x**2 + pos_y**2)
            if dist < 0.5:
                mass, radius, color = GALAXY_MASS * 0.4 / 1, 0.1, YELLOW
            elif dist < 1.5:
                mass, radius, color = GALAXY_MASS * 0.4 / 12, 0.07, WHITE
            else:
                mass, radius, color = GALAXY_MASS * 0.2 / 24, 0.05, GRAY_B
            
            pos1_relative = np.array([pos_x, pos_y, 0])
            pos2_relative = np.array([pos_x, pos_y, 0])
            
            rot_vel1 = self.get_rotation_velocity(pos1_relative, clockwise=True)
            rot_vel2 = self.get_rotation_velocity(pos2_relative, clockwise=True)

            galaxy1_bulbs.append(Bulb(center1 + pos1_relative, mass, INITIAL_VELOCITY + rot_vel1, radius=radius, color=color))
            galaxy2_bulbs.append(Bulb(center2 + pos2_relative, mass, -INITIAL_VELOCITY + rot_vel2, radius=radius, color=color))

        return [galaxy1_bulbs, galaxy2_bulbs]

    def get_rotation_velocity(self, rel_pos, clockwise=True):
        """计算内部旋转速度"""
        if np.linalg.norm(rel_pos) == 0:
            return np.zeros(3)
        direction = np.array([rel_pos[1], -rel_pos[0], 0]) if clockwise else np.array([-rel_pos[1], rel_pos[0], 0])
        return ROTATION_SPEED * direction / (1 + np.linalg.norm(rel_pos))

    def explain_simulation_logic(self, galaxies):
        """动画解释模拟的核心逻辑 (已修复)"""
        bulb = galaxies[0][15]
        text1 = Text("Bulb's Brightness ~ Mass", t2c={"Brightness": YELLOW, "Mass": BLUE}).scale(0.5).to_corner(UR)
        self.play(Write(text1), Indicate(bulb, color=YELLOW))
        self.wait(2)

        force_text = Text("Force is calculated for each bulb", t2c={"Force": RED}).scale(0.5).next_to(text1, DOWN, aligned_edge=RIGHT)
        self.play(Write(force_text))
        
        arrows = VGroup()
        total_force_vec = np.zeros(3)
        all_bulbs = galaxies[0] + galaxies[1]
        for other_bulb in all_bulbs:
            if other_bulb is bulb:
                continue
            
            force_dir = other_bulb.get_center() - bulb.get_center()
            dist_sq = np.sum(force_dir**2)
            if dist_sq < 0.01:
                continue
            
            force_mag = G * bulb.mass * other_bulb.mass / dist_sq
            force_vec = force_mag * force_dir / np.sqrt(dist_sq)
            total_force_vec += force_vec
            
            if other_bulb in galaxies[1][:6] or other_bulb in galaxies[0][:6]:
                arrow = Arrow(bulb.get_center(), other_bulb.get_center(), buff=0.1, stroke_width=2, max_tip_length_to_length_ratio=0.1, color=GREEN)
                arrows.add(arrow)
        
        net_arrow = Arrow(bulb.get_center(), bulb.get_center() + total_force_vec * 0.1, buff=0, color=RED)
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows]), run_time=2)
        self.play(Transform(arrows, net_arrow), run_time=1.5)
        self.wait(1)

        update_text = Text("Position is updated based on this force", t2c={"Position": YELLOW}).scale(0.5).next_to(force_text, DOWN, aligned_edge=RIGHT)
        self.play(Write(update_text))
        
        original_pos = bulb.get_center()
        
        # --- FIX STARTS HERE ---
        simulated_velocity_change = total_force_vec * 0.1
        simulated_pos_change = (bulb.velocity + simulated_velocity_change) * 0.1
        new_pos = original_pos + simulated_pos_change
        
        path = VMobject() # Initialize an empty path
        if np.allclose(original_pos, new_pos, atol=1e-4):
            self.play(Flash(bulb, color=RED, flash_radius=0.2), run_time=2)
        else:
            path = DashedLine(original_pos, new_pos, dash_length=0.05, stroke_width=2)
            self.play(MoveAlongPath(bulb, path), run_time=2)
        
        bulb.move_to(original_pos)
        # --- FIX ENDS HERE ---
        
        self.play(FadeOut(text1, force_text, update_text, arrows, path))
        self.wait(1)

    def run_simulation(self, galaxies, galaxy_group):
        """运行主模拟循环"""
        all_bulbs = galaxies[0] + galaxies[1]

        def update_positions(group, dt):
            for i, bulb1 in enumerate(all_bulbs):
                bulb1.force = np.zeros(3)
                for j, bulb2 in enumerate(all_bulbs):
                    if i == j:
                        continue
                    
                    dir_vec = bulb2.get_center() - bulb1.get_center()
                    dist_sq = np.sum(dir_vec**2)
                    if dist_sq < 0.01:
                        dist_sq = 0.01
                    
                    force_magnitude = G * bulb1.mass * bulb2.mass / dist_sq
                    force_vector = force_magnitude * dir_vec / np.sqrt(dist_sq)
                    bulb1.force += force_vector
            
            for bulb in all_bulbs:
                acceleration = bulb.force / bulb.mass
                bulb.velocity += acceleration * TIME_STEP
                bulb.shift(bulb.velocity * TIME_STEP)
        
        galaxy_group.add_updater(update_positions)
        self.wait(10)
        galaxy_group.remove_updater(update_positions)
        self.wait(2)

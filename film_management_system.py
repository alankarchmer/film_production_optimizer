import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import csv
import pandas as pd
import bcrypt
import smtplib
from email.mime.text import MIMEText

# Constants
LOCATIONS = ["Studio A", "Studio B", "Outdoor Set", "City Street", "Beach"]
TIMES_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
WEATHER_REQUIREMENTS = ["Sunny", "Rainy", "Any"]

# Data classes
@dataclass
class Scene:
    id: int
    actors: List[int]
    crews: List[int]
    equipment: List[int]
    location: str
    estimated_duration: int
    time_of_day: str
    estimated_cost: int
    weather_requirement: str
    dependencies: List[int] = field(default_factory=list)  # List of scene IDs that must precede this scene

@dataclass
class Actor:
    id: int
    name: str
    availability_ranges: List[Tuple[datetime, datetime]]
    salary: int

    def is_available_on(self, date: datetime) -> bool:
        """Check if the actor is available on the given date."""
        return any(start <= date <= end for start, end in self.availability_ranges)

@dataclass
class Crew:
    id: int
    name: str
    role: str
    availability_ranges: List[Tuple[datetime, datetime]]
    daily_rate: int

    def is_available_on(self, date: datetime) -> bool:
        """Check if the crew member is available on the given date."""
        return any(start <= date <= end for start, end in self.availability_ranges)

@dataclass
class Equipment:
    id: int
    name: str
    quantity: int
    availability_ranges: List[Tuple[datetime, datetime]]
    rental_cost_per_day: int

    def is_available_on(self, date: datetime, required_qty: int = 1) -> bool:
        """Check if the required quantity of equipment is available on the given date."""
        # Placeholder for actual availability logic
        return True  # Assume always available for simplicity

@dataclass
class User:
    username: str
    password_hash: bytes
    role: str  # e.g., 'producer', 'assistant', 'admin'

# Scheduler Class
class AdvancedHybridScheduler:
    def __init__(self, scenes, actors, crews, equipment, budget, start_date, population_size=50, generations=1000, learning_rate=0.1):
        self.scenes = scenes
        self.actors = {actor.id: actor for actor in actors}
        self.crews = {crew.id: crew for crew in crews}
        self.equipment = {eq.id: eq for eq in equipment}
        self.budget = budget
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        self.num_days = len(scenes)
        self.population_size = min(population_size, 1000)  # Limit population size to prevent performance issues
        self.generations = generations
        self.learning_rate = learning_rate
        self.phase_weights = np.ones(5) / 5
        self.memory = deque(maxlen=100)
        self.cost_cache = {}
        self.tabu_list = deque(maxlen=20)

    def calculate_cost(self, schedule):
        schedule_tuple = tuple(scene.id for scene in schedule)
        if schedule_tuple in self.cost_cache:
            return self.cost_cache[schedule_tuple]

        cost = 0
        actor_scheduled_days = {}
        crew_scheduled_days = {}
        equipment_usage = {}

        for i, scene in enumerate(schedule):
            date = self.start_date + timedelta(days=i)
            # Actors
            for actor_id in scene.actors:
                if actor_id not in actor_scheduled_days:
                    actor_scheduled_days[actor_id] = set()
                actor_scheduled_days[actor_id].add(date)
            # Crews
            for crew_id in scene.crews:
                if crew_id not in crew_scheduled_days:
                    crew_scheduled_days[crew_id] = set()
                crew_scheduled_days[crew_id].add(date)
            # Equipment
            for eq_id in scene.equipment:
                if eq_id not in equipment_usage:
                    equipment_usage[eq_id] = set()
                equipment_usage[eq_id].add(date)

        # Calculate actor costs
        for actor_id, dates in actor_scheduled_days.items():
            num_days = (max(dates) - min(dates)).days + 1
            cost += self.actors[actor_id].salary * num_days

        # Calculate crew costs
        for crew_id, dates in crew_scheduled_days.items():
            num_days = (max(dates) - min(dates)).days + 1
            cost += self.crews[crew_id].daily_rate * num_days

        # Calculate equipment costs
        for eq_id, dates in equipment_usage.items():
            num_days = len(dates)
            cost += self.equipment[eq_id].rental_cost_per_day * num_days

        # Add estimated scene costs
        for scene in schedule:
            cost += scene.estimated_cost

        self.cost_cache[schedule_tuple] = cost
        return cost

    def is_schedule_feasible(self, schedule):
        """Check if the schedule is feasible considering actor/crew availability, equipment availability, dependencies, and budget."""
        for i, scene in enumerate(schedule):
            date = self.start_date + timedelta(days=i)
            # Check actor availability
            for actor_id in scene.actors:
                actor = self.actors[actor_id]
                if not actor.is_available_on(date):
                    return False
            # Check crew availability
            for crew_id in scene.crews:
                crew = self.crews[crew_id]
                if not crew.is_available_on(date):
                    return False
            # Check equipment availability
            for eq_id in scene.equipment:
                equipment = self.equipment[eq_id]
                if not equipment.is_available_on(date):
                    return False
            # Check dependencies
            for dep_id in scene.dependencies:
                dep_scene = next((s for s in self.scenes if s.id == dep_id), None)
                if dep_scene:
                    dep_index = schedule.index(dep_scene)
                    if dep_index >= i:
                        return False  # Dependency scene is not scheduled before current scene
        # Check budget constraint
        if self.calculate_cost(schedule) > self.budget:
            return False
        return True

    def detect_conflicts(self, schedule):
        """Detect scheduling conflicts such as overlapping actor or crew assignments."""
        conflicts = []
        actor_schedule = {}
        crew_schedule = {}

        for i, scene in enumerate(schedule):
            date = self.start_date + timedelta(days=i)
            # Actors
            for actor_id in scene.actors:
                if actor_id in actor_schedule and date in actor_schedule[actor_id]:
                    conflicts.append(f"Actor {actor_id} has a conflict on {date.strftime('%Y-%m-%d')}")
                else:
                    actor_schedule.setdefault(actor_id, set()).add(date)
            # Crews
            for crew_id in scene.crews:
                if crew_id in crew_schedule and date in crew_schedule[crew_id]:
                    conflicts.append(f"Crew {crew_id} has a conflict on {date.strftime('%Y-%m-%d')}")
                else:
                    crew_schedule.setdefault(crew_id, set()).add(date)
        return conflicts

    def cheng_heuristic(self):
        schedule = self.scenes.copy()
        random.shuffle(schedule)
        return schedule

    def genetic_crossover(self, parent1, parent2):
        if len(parent1) < 2:
            return parent1
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point]
        child.extend([scene for scene in parent2 if scene not in child])
        return child

    def simulated_annealing(self, schedule, temperature=100, cooling_rate=0.995):
        current_schedule = schedule.copy()
        current_cost = self.calculate_cost(current_schedule)
        best_schedule = current_schedule
        best_cost = current_cost

        while temperature > 0.1:
            if len(schedule) < 2:
                break
            i, j = random.sample(range(len(schedule)), 2)
            new_schedule = current_schedule.copy()
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

            if self.is_schedule_feasible(new_schedule):
                new_cost = self.calculate_cost(new_schedule)

                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
                    current_schedule = new_schedule
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_schedule = current_schedule
                        best_cost = current_cost

            temperature *= cooling_rate

        return best_schedule

    def local_search(self, schedule):
        improved = True
        while improved:
            improved = False
            for i in range(len(schedule)):
                for j in range(i + 1, len(schedule)):
                    new_schedule = schedule.copy()
                    new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
                    if self.is_schedule_feasible(new_schedule) and self.calculate_cost(new_schedule) < self.calculate_cost(schedule):
                        schedule = new_schedule
                        improved = True
                        break
                if improved:
                    break
        return schedule

    def adjust_for_dependencies(self, schedule):
        """Ensure that all dependencies are respected in the schedule."""
        schedule = schedule.copy()
        for scene in schedule:
            for dep_id in scene.dependencies:
                dep_scene = next((s for s in schedule if s.id == dep_id), None)
                if dep_scene and schedule.index(dep_scene) > schedule.index(scene):
                    # Swap scenes to respect dependency
                    schedule.remove(dep_scene)
                    schedule.insert(schedule.index(scene), dep_scene)
        return schedule

    def update_phase_weights(self, phase, improvement):
        self.phase_weights[phase] += self.learning_rate * improvement
        self.phase_weights = np.clip(self.phase_weights, 0.1, 1.0)
        self.phase_weights /= np.sum(self.phase_weights)

    def optimize_with_constraints(self):
        if not self.scenes:
            raise ValueError("No scenes to schedule")

        # Generate an initial population of feasible schedules
        population = []
        attempts = 0
        max_attempts = self.population_size * 5  # Limit the number of attempts to prevent infinite loops
        while len(population) < self.population_size and attempts < max_attempts:
            schedule = self.cheng_heuristic()
            schedule = self.adjust_for_dependencies(schedule)
            if self.is_schedule_feasible(schedule):
                population.append(schedule)
            attempts += 1

        if not population:
            raise ValueError("No feasible schedule found in initial population.")

        best_schedule = min(population, key=self.calculate_cost)
        best_cost = self.calculate_cost(best_schedule)

        # Proceed with optimization as before
        for _ in range(self.generations):
            phase = np.random.choice(5, p=self.phase_weights)

            if phase == 0:  # Cheng's heuristic
                new_schedule = self.cheng_heuristic()
                new_schedule = self.adjust_for_dependencies(new_schedule)
            elif phase == 1 and len(population) >= 2:  # Genetic algorithm
                parents = random.sample(population, 2)
                new_schedule = self.genetic_crossover(parents[0], parents[1])
                new_schedule = self.adjust_for_dependencies(new_schedule)
            elif phase == 2:  # Simulated annealing
                parent = random.choice(population)
                new_schedule = self.simulated_annealing(parent)
                new_schedule = self.adjust_for_dependencies(new_schedule)
            elif phase == 3:  # Local search
                parent = random.choice(population)
                new_schedule = self.local_search(parent)
                new_schedule = self.adjust_for_dependencies(new_schedule)
            else:  # Conflict resolution or other
                parent = random.choice(population)
                new_schedule = self.adjust_for_dependencies(parent)
                # Additional conflict resolution logic can be added here

            if self.is_schedule_feasible(new_schedule):
                new_cost = self.calculate_cost(new_schedule)

                if new_cost < best_cost:
                    improvement = (best_cost - new_cost) / best_cost
                    self.update_phase_weights(phase, improvement)
                    best_schedule = new_schedule
                    best_cost = new_cost

                population[random.randint(0, self.population_size - 1)] = new_schedule
                self.memory.append((new_schedule, new_cost))

        return best_schedule, best_cost

    def group_scenes_by_location(self):
        location_groups = {}
        for scene in self.scenes:
            if scene.location not in location_groups:
                location_groups[scene.location] = []
            location_groups[scene.location].append(scene)
        return location_groups

    def group_scenes_by_time(self, scenes):
        time_groups = {}
        for scene in scenes:
            if scene.time_of_day not in time_groups:
                time_groups[scene.time_of_day] = []
            time_groups[scene.time_of_day].append(scene)
        return time_groups

    def optimize(self):
        location_groups = self.group_scenes_by_location()
        final_schedule = []
        total_cost = 0

        for location, location_scenes in location_groups.items():
            time_groups = self.group_scenes_by_time(location_scenes)
            location_schedule = []

            for time, time_scenes in time_groups.items():
                self.scenes = time_scenes  # Temporarily set scenes for optimization
                try:
                    time_schedule, time_cost = self.optimize_with_constraints()
                    location_schedule.extend(time_schedule)
                    total_cost += time_cost
                except ValueError as e:
                    continue  # Skip if no feasible schedule is found for this group

            final_schedule.extend(location_schedule)

        final_schedule = self.adjust_for_dependencies(final_schedule)
        return final_schedule, total_cost

# User Authentication Placeholder
class UserManager:
    def __init__(self):
        self.users: Dict[str, User] = {}

    def register_user(self, username, password, role):
        if username in self.users:
            raise ValueError("Username already exists")
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.users[username] = User(username=username, password_hash=password_hash, role=role)

    def authenticate_user(self, username, password):
        user = self.users.get(username)
        if not user:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), user.password_hash)

# GUI Class
class TalentSchedulingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Talent Scheduling Optimizer")
        master.geometry("1600x1000")
        master.minsize(1000, 700)

        # Initialize User Manager
        self.user_manager = UserManager()
        self.current_user = None

        # Create main frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a paned window
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left frame (inputs and data display)
        self.left_frame = ttk.Frame(self.paned_window, width=500)
        self.paned_window.add(self.left_frame)

        # Right frame (results and graph)
        self.right_frame = ttk.Frame(self.paned_window, width=1100)
        self.paned_window.add(self.right_frame)

        # Initialize data containers
        self.scenes: List[Scene] = []
        self.actors: List[Actor] = []
        self.crews: List[Crew] = []
        self.equipment: List[Equipment] = []
        self.schedule: List[Scene] = []
        self.total_cost: int = 0

        # Create widgets
        self.create_login_widgets()
        self.create_input_widgets(self.left_frame)
        self.create_data_display_widgets(self.left_frame)
        self.create_result_widgets(self.right_frame)

        # Connect the closing event
        master.protocol("WM_DELETE_WINDOW", self._quit)

    def create_login_widgets(self):
        """Create login interface."""
        self.login_frame = ttk.Frame(self.left_frame)
        self.login_frame.pack(pady=20)

        ttk.Label(self.login_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.username_entry = ttk.Entry(self.login_frame, width=20)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.login_frame, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.password_entry = ttk.Entry(self.login_frame, width=20, show="*")
        self.password_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self.login_frame, text="Login", command=self.login).grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(self.login_frame, text="Register", command=self.register).grid(row=3, column=0, columnspan=2, pady=5)

    def login(self):
        """Handle user login."""
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.user_manager.authenticate_user(username, password):
            self.current_user = self.user_manager.users[username]
            messagebox.showinfo("Login Successful", f"Welcome, {username}!")
            self.login_frame.destroy()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def register(self):
        """Handle new user registration."""
        username = self.username_entry.get()
        password = self.password_entry.get()
        role = 'producer'  # Default role; can be modified as needed
        try:
            self.user_manager.register_user(username, password, role)
            messagebox.showinfo("Registration Successful", "You can now log in with your credentials.")
        except ValueError as e:
            messagebox.showerror("Registration Failed", str(e))

    def create_input_widgets(self, parent):
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input Parameters")
        input_frame.pack(fill=tk.X, pady=5, padx=5)

        # Use grid layout for better control
        ttk.Label(input_frame, text="Number of Actors:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_actors_entry = ttk.Entry(input_frame, width=10)
        self.num_actors_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(input_frame, text="Number of Crews:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_crews_entry = ttk.Entry(input_frame, width=10)
        self.num_crews_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(input_frame, text="Number of Equipment:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_equipment_entry = ttk.Entry(input_frame, width=10)
        self.num_equipment_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(input_frame, text="Number of Scenes:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.num_scenes_entry = ttk.Entry(input_frame, width=10)
        self.num_scenes_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(input_frame, text="Budget:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.budget_entry = ttk.Entry(input_frame, width=10)
        self.budget_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.start_date_entry = ttk.Entry(input_frame, width=10)
        self.start_date_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Button(input_frame, text="Generate Random Data", command=self.generate_random_data).grid(row=6, column=0, columnspan=2, pady=10)

        # Configure grid to expand properly
        for i in range(7):
            input_frame.grid_rowconfigure(i, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)

    def create_data_display_widgets(self, parent):
        # Data display section
        data_frame = ttk.Frame(parent)
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Scenes
        ttk.Label(data_frame, text="Scenes").pack()
        self.scene_text = scrolledtext.ScrolledText(data_frame, height=10, width=60)
        self.scene_text.pack(fill=tk.BOTH, expand=True)

        # Actors
        ttk.Label(data_frame, text="Actors").pack()
        self.actor_text = scrolledtext.ScrolledText(data_frame, height=10, width=60)
        self.actor_text.pack(fill=tk.BOTH, expand=True)

        # Crews
        ttk.Label(data_frame, text="Crews").pack()
        self.crew_text = scrolledtext.ScrolledText(data_frame, height=10, width=60)
        self.crew_text.pack(fill=tk.BOTH, expand=True)

        # Equipment
        ttk.Label(data_frame, text="Equipment").pack()
        self.equipment_text = scrolledtext.ScrolledText(data_frame, height=10, width=60)
        self.equipment_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(parent, text="Run Scheduler", command=self.run_scheduler).pack(pady=10)
        ttk.Button(parent, text="Export Schedule", command=self.export_schedule).pack(pady=5)
        ttk.Button(parent, text="Import Schedule", command=self.import_schedule).pack(pady=5)

    def create_result_widgets(self, parent):
        # Results section with scrollbar
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add vertical scrollbar for result text
        result_scroll_y = ttk.Scrollbar(result_frame, orient=tk.VERTICAL)
        result_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(result_frame, wrap=tk.WORD, width=80, height=30, yscrollcommand=result_scroll_y.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll_y.config(command=self.result_text.yview)

        ttk.Label(parent, text="Scheduling Results").pack()

        # Matplotlib figure for visualization
        fig_frame = ttk.Frame(parent)
        fig_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 8), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        plot_widget = self.plot_canvas.get_tk_widget()
        plot_widget.pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, fig_frame)
        self.toolbar.update()
        self.plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def generate_random_data(self):
        try:
            num_actors = int(self.num_actors_entry.get())
            num_crews = int(self.num_crews_entry.get())
            num_equipment = int(self.num_equipment_entry.get())
            num_scenes = int(self.num_scenes_entry.get())
            budget = float(self.budget_entry.get())
            start_date = datetime.strptime(self.start_date_entry.get(), "%Y-%m-%d")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
            return

        # Set random seed for reproducibility
        random.seed(42)

        schedule_length = num_scenes

        # Generate actors with full availability
        self.actors = []
        for i in range(num_actors):
            salary = random.randint(1000, 10000)
            # Make actors available throughout the scheduling period
            availability_ranges = [(start_date, start_date + timedelta(days=schedule_length - 1))]
            self.actors.append(Actor(id=i, name=f"Actor_{i}", availability_ranges=availability_ranges, salary=salary))

        # Generate crews with full availability
        self.crews = []
        for i in range(num_crews):
            daily_rate = random.randint(1500, 12000)
            availability_ranges = [(start_date, start_date + timedelta(days=schedule_length - 1))]
            self.crews.append(Crew(id=i, name=f"Crew_{i}", role=random.choice(["Director", "Camera Operator", "Sound Engineer", "Lighting Technician"]),
                                   availability_ranges=availability_ranges, daily_rate=daily_rate))

        # Generate equipment
        self.equipment = []
        for i in range(num_equipment):
            rental_cost = random.randint(500, 5000)
            availability_ranges = [(start_date, start_date + timedelta(days=schedule_length - 1))]
            self.equipment.append(Equipment(id=i, name=f"Equipment_{i}", quantity=random.randint(1, 5),
                                           availability_ranges=availability_ranges, rental_cost_per_day=rental_cost))

        # Generate scenes with dependencies
        self.scenes = []
        for i in range(num_scenes):
            actors_in_scene = random.sample(range(num_actors), random.randint(1, min(5, num_actors)))
            crews_in_scene = random.sample(range(num_crews), random.randint(1, min(3, num_crews)))
            equipment_in_scene = random.sample(range(num_equipment), random.randint(1, min(3, num_equipment)))
            location = random.choice(LOCATIONS)
            estimated_duration = random.randint(1, 8)  # hours
            tod = random.choice(TIMES_OF_DAY)
            estimated_cost = random.randint(5000, 50000)
            weather_req = random.choice(WEATHER_REQUIREMENTS)
            dependencies = []
            if i > 0:
                # Randomly assign dependencies
                if random.random() < 0.3:
                    dep_scene = random.randint(0, i - 1)
                    dependencies.append(dep_scene)
            self.scenes.append(Scene(id=i, actors=actors_in_scene, crews=crews_in_scene, equipment=equipment_in_scene,
                                     location=location, estimated_duration=estimated_duration, time_of_day=tod,
                                     estimated_cost=estimated_cost, weather_requirement=weather_req, dependencies=dependencies))

        self.display_generated_data()
        messagebox.showinfo("Data Generated", f"Generated {num_scenes} scenes, {num_actors} actors, {num_crews} crews, and {num_equipment} equipment.")

    def display_generated_data(self):
        # Display Scenes
        self.scene_text.delete('1.0', tk.END)
        for scene in self.scenes:
            deps = ', '.join(map(str, scene.dependencies)) if scene.dependencies else 'None'
            self.scene_text.insert(tk.END, f"Scene {scene.id}: {scene.location}, {scene.time_of_day}, {scene.weather_requirement}\n")
            self.scene_text.insert(tk.END, f"    Actors: {', '.join(map(str, scene.actors))}\n")
            self.scene_text.insert(tk.END, f"    Crews: {', '.join(map(str, scene.crews))}\n")
            self.scene_text.insert(tk.END, f"    Equipment: {', '.join(map(str, scene.equipment))}\n")
            self.scene_text.insert(tk.END, f"    Dependencies: {deps}\n")
            self.scene_text.insert(tk.END, f"    Duration: {scene.estimated_duration} hours, Cost: ${scene.estimated_cost}\n\n")

        # Display Actors
        self.actor_text.delete('1.0', tk.END)
        for actor in self.actors:
            availability_str = ', '.join([f"{start.date()} to {end.date()}" for start, end in actor.availability_ranges])
            self.actor_text.insert(tk.END, f"Actor {actor.id}: ${actor.salary}/day\n")
            self.actor_text.insert(tk.END, f"    Availability: {availability_str}\n\n")

        # Display Crews
        self.crew_text.delete('1.0', tk.END)
        for crew in self.crews:
            availability_str = ', '.join([f"{start.date()} to {end.date()}" for start, end in crew.availability_ranges])
            self.crew_text.insert(tk.END, f"Crew {crew.id}: {crew.role}, ${crew.daily_rate}/day\n")
            self.crew_text.insert(tk.END, f"    Availability: {availability_str}\n\n")

        # Display Equipment
        self.equipment_text.delete('1.0', tk.END)
        for eq in self.equipment:
            availability_str = ', '.join([f"{start.date()} to {end.date()}" for start, end in eq.availability_ranges])
            self.equipment_text.insert(tk.END, f"Equipment {eq.id}: {eq.name}, Quantity: {eq.quantity}, ${eq.rental_cost_per_day}/day\n")
            self.equipment_text.insert(tk.END, f"    Availability: {availability_str}\n\n")

    def run_scheduler(self):
        if not self.scenes or not self.actors or not self.crews or not self.equipment:
            messagebox.showerror("Error", "Please generate random data first.")
            return

        try:
            budget = float(self.budget_entry.get())
            start_date = self.start_date_entry.get()

            scheduler = AdvancedHybridScheduler(
                scenes=self.scenes,
                actors=self.actors,
                crews=self.crews,
                equipment=self.equipment,
                budget=budget,
                start_date=start_date
            )
            self.schedule, self.total_cost = scheduler.optimize()

            # Detect conflicts
            conflicts = scheduler.detect_conflicts(self.schedule)
            if conflicts:
                conflict_message = "\n".join(conflicts)
                messagebox.showwarning("Conflicts Detected", conflict_message)

            self.display_results()
            self.visualize_schedule()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Detailed error: {e}")

    def display_results(self):
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, f"Optimized Schedule:\n\n")

        for day, scene in enumerate(self.schedule, 1):
            date = self.start_date + timedelta(days=day - 1)
            self.result_text.insert(tk.END, f"Day {day} ({date.strftime('%Y-%m-%d')}): Scene {scene.id} - {scene.location}, {scene.time_of_day}\n")
            self.result_text.insert(tk.END, f"    Actors: {', '.join([str(actor_id) for actor_id in scene.actors])}\n")
            self.result_text.insert(tk.END, f"    Crews: {', '.join([str(crew_id) for crew_id in scene.crews])}\n")
            self.result_text.insert(tk.END, f"    Equipment: {', '.join([str(eq_id) for eq_id in scene.equipment])}\n")
            self.result_text.insert(tk.END, f"    Weather Requirement: {scene.weather_requirement}\n")
            self.result_text.insert(tk.END, f"    Dependencies: {', '.join(map(str, scene.dependencies)) if scene.dependencies else 'None'}\n")
            self.result_text.insert(tk.END, f"    Estimated Duration: {scene.estimated_duration} hours\n")
            self.result_text.insert(tk.END, f"    Estimated Cost: ${scene.estimated_cost}\n\n")

        # Detailed Budget Breakdown
        self.result_text.insert(tk.END, f"\nTotal Scheduling Cost: ${self.total_cost}\n")

    def visualize_schedule(self):
        self.ax.clear()

        locations = list(set(scene.location for scene in self.schedule))
        location_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(locations)))
        color_map = dict(zip(locations, location_colors))

        for day, scene in enumerate(self.schedule):
            self.ax.bar(day, scene.estimated_duration, color=color_map[scene.location], align='center', alpha=0.7)
            self.ax.text(day, scene.estimated_duration / 2, f"S{scene.id}", ha='center', va='center', fontsize=8)

        self.ax.set_xlabel("Shooting Day")
        self.ax.set_ylabel("Estimated Duration (hours)")
        self.ax.set_title("Optimized Shooting Schedule")

        dates = [(self.start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(self.schedule))]
        self.ax.set_xticks(range(len(self.schedule)))
        self.ax.set_xticklabels(dates, rotation=45, ha='right')

        # Add a legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7) for color in color_map.values()]
        self.ax.legend(legend_elements, locations, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

        plt.tight_layout()
        self.plot_canvas.draw()

    def export_schedule(self):
        if not self.schedule:
            messagebox.showerror("Error", "No schedule to export.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv"),
                                                           ("Excel files", "*.xlsx"),
                                                           ("PDF files", "*.pdf")])
        if not filepath:
            return  # User cancelled

        try:
            if filepath.endswith('.csv'):
                self.export_schedule_to_csv(filepath)
            elif filepath.endswith('.xlsx'):
                self.export_schedule_to_excel(filepath)
            elif filepath.endswith('.pdf'):
                self.export_schedule_to_pdf(filepath)
            messagebox.showinfo("Export Successful", f"Schedule exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred during export: {str(e)}")

    def export_schedule_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Day', 'Date', 'Scene ID', 'Location', 'Time of Day', 'Actors', 'Crews', 'Equipment',
                             'Weather Requirement', 'Dependencies', 'Duration (hours)', 'Estimated Cost'])
            for day, scene in enumerate(self.schedule, 1):
                date = self.start_date + timedelta(days=day - 1)
                writer.writerow([
                    day,
                    date.strftime('%Y-%m-%d'),
                    scene.id,
                    scene.location,
                    scene.time_of_day,
                    ','.join(map(str, scene.actors)),
                    ','.join(map(str, scene.crews)),
                    ','.join(map(str, scene.equipment)),
                    scene.weather_requirement,
                    ','.join(map(str, scene.dependencies)) if scene.dependencies else 'None',
                    scene.estimated_duration,
                    scene.estimated_cost
                ])

    def export_schedule_to_excel(self, filepath):
        data = []
        for day, scene in enumerate(self.schedule, 1):
            date = self.start_date + timedelta(days=day - 1)
            data.append({
                'Day': day,
                'Date': date.strftime('%Y-%m-%d'),
                'Scene ID': scene.id,
                'Location': scene.location,
                'Time of Day': scene.time_of_day,
                'Actors': ','.join(map(str, scene.actors)),
                'Crews': ','.join(map(str, scene.crews)),
                'Equipment': ','.join(map(str, scene.equipment)),
                'Weather Requirement': scene.weather_requirement,
                'Dependencies': ','.join(map(str, scene.dependencies)) if scene.dependencies else 'None',
                'Duration (hours)': scene.estimated_duration,
                'Estimated Cost': scene.estimated_cost
            })
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)

    def export_schedule_to_pdf(self, filepath):
        try:
            from fpdf import FPDF
        except ImportError:
            messagebox.showerror("Import Error", "Please install fpdf library to export as PDF.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for day, scene in enumerate(self.schedule, 1):
            date = self.start_date + timedelta(days=day - 1)
            pdf.multi_cell(0, 10, f"Day {day} ({date.strftime('%Y-%m-%d')}): Scene {scene.id} - {scene.location}, {scene.time_of_day}")
            pdf.multi_cell(0, 10, f"    Actors: {', '.join([str(actor_id) for actor_id in scene.actors])}")
            pdf.multi_cell(0, 10, f"    Crews: {', '.join([str(crew_id) for crew_id in scene.crews])}")
            pdf.multi_cell(0, 10, f"    Equipment: {', '.join([str(eq_id) for eq_id in scene.equipment])}")
            pdf.multi_cell(0, 10, f"    Weather Requirement: {scene.weather_requirement}")
            pdf.multi_cell(0, 10, f"    Dependencies: {', '.join(map(str, scene.dependencies)) if scene.dependencies else 'None'}")
            pdf.multi_cell(0, 10, f"    Estimated Duration: {scene.estimated_duration} hours")
            pdf.multi_cell(0, 10, f"    Estimated Cost: ${scene.estimated_cost}\n")
        pdf.output(filepath)

    def import_schedule(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"),
                                                         ("Excel files", "*.xlsx")])
        if not filepath:
            return  # User cancelled

        try:
            if filepath.endswith('.csv'):
                self.import_schedule_from_csv(filepath)
            elif filepath.endswith('.xlsx'):
                self.import_schedule_from_excel(filepath)
            messagebox.showinfo("Import Successful", f"Schedule imported from {filepath}")
            self.display_results()
            self.visualize_schedule()
        except Exception as e:
            messagebox.showerror("Import Failed", f"An error occurred during import: {str(e)}")

    def import_schedule_from_csv(self, filepath):
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            imported_schedule = []
            for row in reader:
                scene_id = int(row['Scene ID'])
                location = row['Location']
                time_of_day = row['Time of Day']
                weather_requirement = row['Weather Requirement']
                dependencies = [int(dep) for dep in row['Dependencies'].split(',')] if row['Dependencies'] != 'None' else []
                duration = int(row['Duration (hours)'])
                cost = int(row['Estimated Cost'])
                actors = [int(a) for a in row['Actors'].split(',')]
                crews = [int(c) for c in row['Crews'].split(',')]
                equipment = [int(e) for e in row['Equipment'].split(',')]
                scene = Scene(id=scene_id, actors=actors, crews=crews, equipment=equipment,
                              location=location, time_of_day=time_of_day,
                              estimated_cost=cost, weather_requirement=weather_requirement,
                              dependencies=dependencies, estimated_duration=duration)
                imported_schedule.append(scene)
            self.schedule = imported_schedule
            # Recalculate total cost
            self.total_cost = sum(scene.estimated_cost for scene in self.schedule)

    def import_schedule_from_excel(self, filepath):
        df = pd.read_excel(filepath)
        imported_schedule = []
        for _, row in df.iterrows():
            scene_id = int(row['Scene ID'])
            location = row['Location']
            time_of_day = row['Time of Day']
            weather_requirement = row['Weather Requirement']
            dependencies = [int(dep) for dep in row['Dependencies'].split(',')] if row['Dependencies'] != 'None' else []
            duration = int(row['Duration (hours)'])
            cost = int(row['Estimated Cost'])
            actors = [int(a) for a in row['Actors'].split(',')]
            crews = [int(c) for c in row['Crews'].split(',')]
            equipment = [int(e) for e in row['Equipment'].split(',')]
            scene = Scene(id=scene_id, actors=actors, crews=crews, equipment=equipment,
                          location=location, time_of_day=time_of_day,
                          estimated_cost=cost, weather_requirement=weather_requirement,
                          dependencies=dependencies, estimated_duration=duration)
            imported_schedule.append(scene)
        self.schedule = imported_schedule
        # Recalculate total cost
        self.total_cost = sum(scene.estimated_cost for scene in self.schedule)

    def _quit(self):
        self.master.quit()
        self.master.destroy()

def main():
    root = tk.Tk()
    root.geometry("1600x1000")
    app = TalentSchedulingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
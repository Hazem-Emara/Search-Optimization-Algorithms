"""
=============================================================================
  AI SEARCH & OPTIMIZATION ALGORITHMS — Complete Implementation
  Covers: BFS, DFS, UCS, Greedy, A*, Genetic Algorithm
  Each algorithm: Iterative + Recursive + Real-World Application
=============================================================================
"""

import heapq
import random
import math
from collections import deque, defaultdict

# =============================================================================
# 1.  BFS — BREADTH-FIRST SEARCH
# =============================================================================

# ── 1A. BFS Iterative ────────────────────────────────────────────────────────
def bfs_iterative(graph, start, goal):
    """BFS using an explicit queue."""
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None

# ── 1B. BFS Recursive ────────────────────────────────────────────────────────
def bfs_recursive(graph, queue, visited, goal):
    """BFS implemented recursively (queue passed each call)."""
    if not queue:
        return None
    path = queue.popleft()
    node = path[-1]
    if node == goal:
        return path
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(path + [neighbor])
    return bfs_recursive(graph, queue, visited, goal)

def bfs_recursive_start(graph, start, goal):
    queue = deque([[start]])
    return bfs_recursive(graph, queue, {start}, goal)

# ── 1C. Real-World: Social Network — Degrees of Separation ──────────────────
def degrees_of_separation(network, person_a, person_b):
    """
    Find the shortest connection path between two people
    in a social network (e.g., LinkedIn, Facebook).
    Returns the connection chain and degree count.
    """
    if person_a == person_b:
        return [person_a], 0
    queue = deque([[person_a]])
    visited = {person_a}
    while queue:
        path = queue.popleft()
        person = path[-1]
        for friend in network.get(person, []):
            if friend not in visited:
                new_path = path + [friend]
                if friend == person_b:
                    return new_path, len(new_path) - 1
                visited.add(friend)
                queue.append(new_path)
    return None, -1

# ── 1D. Real-World: Web Crawler — Link Discovery ────────────────────────────
def web_crawler_bfs(link_map, start_url, max_pages=10):
    """
    Simulate a BFS-based web crawler that discovers pages
    level by level (like Google's crawler).
    Returns visited pages in crawl order.
    """
    visited = []
    queue = deque([start_url])
    seen = {start_url}
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        visited.append(url)
        for link in link_map.get(url, []):
            if link not in seen:
                seen.add(link)
                queue.append(link)
    return visited


# =============================================================================
# 2.  DFS — DEPTH-FIRST SEARCH
# =============================================================================

# ── 2A. DFS Iterative ────────────────────────────────────────────────────────
def dfs_iterative(graph, start, goal):
    """DFS using an explicit stack."""
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

# ── 2B. DFS Recursive ────────────────────────────────────────────────────────
def dfs_recursive(graph, node, goal, visited=None, path=None):
    """Classic recursive DFS."""
    if visited is None: visited = set()
    if path is None: path = []
    visited.add(node)
    path = path + [node]
    if node == goal:
        return path
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            result = dfs_recursive(graph, neighbor, goal, visited, path)
            if result:
                return result
    return None

# ── 2C. Real-World: Maze Solver ──────────────────────────────────────────────
def solve_maze_dfs(maze, start, end):
    """
    Solve a 2D maze using DFS backtracking.
    maze: 2D list  (0 = open path, 1 = wall)
    Returns the path from start to end or None.
    """
    rows, cols = len(maze), len(maze[0])
    visited = set()

    def dfs(r, c, path):
        if (r, c) == end:
            return path + [(r, c)]
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or maze[r][c] == 1 or (r, c) in visited):
            return None
        visited.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            result = dfs(r+dr, c+dc, path + [(r,c)])
            if result:
                return result
        return None

    return dfs(start[0], start[1], [])

# ── 2D. Real-World: Dependency Resolver ─────────────────────────────────────
def topological_sort_dfs(dependencies):
    """
    Resolve package/task dependencies using DFS topological sort.
    E.g., npm install, Makefile, course prerequisites.
    Returns installation order or detects circular dependency.
    """
    visited = set()
    order = []
    in_progress = set()

    def dfs(pkg):
        if pkg in in_progress:
            raise ValueError(f"Circular dependency detected at: {pkg}")
        if pkg in visited:
            return
        in_progress.add(pkg)
        for dep in dependencies.get(pkg, []):
            dfs(dep)
        in_progress.remove(pkg)
        visited.add(pkg)
        order.append(pkg)

    for pkg in dependencies:
        dfs(pkg)
    return order


# =============================================================================
# 3.  UCS — UNIFORM COST SEARCH
# =============================================================================

# ── 3A. UCS Iterative ────────────────────────────────────────────────────────
def ucs_iterative(graph, start, goal):
    """
    UCS using a min-heap priority queue.
    graph: dict  {node: [(neighbor, cost), ...]}
    Returns (total_cost, path).
    """
    heap = [(0, start, [start])]
    visited = {}
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited[node] = cost
        if node == goal:
            return cost, path
        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))
    return float('inf'), []

# ── 3B. UCS Recursive ────────────────────────────────────────────────────────
def ucs_recursive(graph, heap, visited, goal):
    """UCS implemented recursively."""
    if not heap:
        return float('inf'), []
    cost, node, path = heapq.heappop(heap)
    if node in visited:
        return ucs_recursive(graph, heap, visited, goal)
    visited[node] = cost
    if node == goal:
        return cost, path
    for neighbor, edge_cost in graph.get(node, []):
        if neighbor not in visited:
            heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))
    return ucs_recursive(graph, heap, visited, goal)

def ucs_recursive_start(graph, start, goal):
    heap = [(0, start, [start])]
    return ucs_recursive(graph, heap, {}, goal)

# ── 3C. Real-World: Cheapest Flight Route ────────────────────────────────────
def cheapest_flight(flights, origin, destination):
    """
    Find the cheapest flight route between two cities.
    Simulates Google Flights / Skyscanner cost optimization.
    flights: dict  {city: [(next_city, price), ...]}
    """
    cost, path = ucs_iterative(flights, origin, destination)
    if path:
        route = " -> ".join(path)
        return f"Cheapest route: {route}  |  Total cost: ${cost}"
    return "No route found"

# ── 3D. Real-World: Network Packet Routing ───────────────────────────────────
def network_routing(routers, source, destination):
    """
    Find the lowest-latency path through a network.
    Used in OSPF (Open Shortest Path First) routing protocol.
    routers: dict  {router: [(next_router, latency_ms), ...]}
    Returns (latency_ms, path).
    """
    return ucs_iterative(routers, source, destination)


# =============================================================================
# 4.  GREEDY BEST-FIRST SEARCH
# =============================================================================

# ── 4A. Greedy Iterative ─────────────────────────────────────────────────────
def greedy_iterative(graph, start, goal, heuristic):
    """
    Greedy search: always expands node with lowest h(n).
    heuristic: dict or callable  {node: estimated_cost_to_goal}
    """
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    heap = [(h(start), start, [start])]
    visited = set()
    while heap:
        _, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(heap, (h(neighbor), neighbor, path + [neighbor]))
    return None

# ── 4B. Greedy Recursive ─────────────────────────────────────────────────────
def greedy_recursive(graph, heap, visited, goal, heuristic):
    """Greedy Best-First implemented recursively."""
    if not heap:
        return None
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    _, node, path = heapq.heappop(heap)
    if node in visited:
        return greedy_recursive(graph, heap, visited, goal, heuristic)
    visited.add(node)
    if node == goal:
        return path
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            heapq.heappush(heap, (h(neighbor), neighbor, path + [neighbor]))
    return greedy_recursive(graph, heap, visited, goal, heuristic)

def greedy_recursive_start(graph, start, goal, heuristic):
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    heap = [(h(start), start, [start])]
    return greedy_recursive(graph, heap, set(), goal, heuristic)

# ── 4C. Real-World: GPS Navigation (Fast Approximate) ────────────────────────
def gps_greedy(road_map, start, destination, straight_line_dist):
    """
    Fast GPS navigation using Greedy search.
    Uses straight-line distance as heuristic.
    Finds a quick route (not necessarily the absolute shortest).
    road_map: {city: [neighbor_city, ...]}
    straight_line_dist: {city: distance_to_destination}
    """
    path = greedy_iterative(road_map, start, destination, straight_line_dist)
    if path:
        return " -> ".join(path)
    return "No route found"

# ── 4D. Real-World: Auto-Complete Suggestion ─────────────────────────────────
def autocomplete_greedy(word_graph, partial_word, target_word, proximity_scores):
    """
    Simulate auto-complete using Greedy search through a
    word-similarity graph. Picks the most likely next suggestion.
    proximity_scores: {word: score_to_target}
    """
    path = greedy_iterative(word_graph, partial_word, target_word, proximity_scores)
    return path if path else []


# =============================================================================
# 5.  A* SEARCH ALGORITHM
# =============================================================================

# ── 5A. A* Iterative ─────────────────────────────────────────────────────────
def astar_iterative(graph, start, goal, heuristic):
    """
    A* Search: f(n) = g(n) + h(n)
    graph: {node: [(neighbor, cost), ...]}
    heuristic: dict or callable
    Returns (total_cost, path).
    """
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    heap = [(h(start), 0, start, [start])]
    g_costs = {start: 0}
    while heap:
        f, g, node, path = heapq.heappop(heap)
        if node == goal:
            return g, path
        if g > g_costs.get(node, float('inf')):
            continue
        for neighbor, cost in graph.get(node, []):
            new_g = g + cost
            if new_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = new_g
                f_val = new_g + h(neighbor)
                heapq.heappush(heap, (f_val, new_g, neighbor, path + [neighbor]))
    return float('inf'), []

# ── 5B. A* Recursive ─────────────────────────────────────────────────────────
def astar_recursive(heap, g_costs, goal, graph, heuristic):
    """A* implemented recursively."""
    if not heap:
        return float('inf'), []
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    f, g, node, path = heapq.heappop(heap)
    if node == goal:
        return g, path
    if g > g_costs.get(node, float('inf')):
        return astar_recursive(heap, g_costs, goal, graph, heuristic)
    for neighbor, cost in graph.get(node, []):
        new_g = g + cost
        if new_g < g_costs.get(neighbor, float('inf')):
            g_costs[neighbor] = new_g
            heapq.heappush(heap, (new_g + h(neighbor), new_g, neighbor, path + [neighbor]))
    return astar_recursive(heap, g_costs, goal, graph, heuristic)

def astar_recursive_start(graph, start, goal, heuristic):
    h = heuristic if callable(heuristic) else lambda n: heuristic.get(n, float('inf'))
    heap = [(h(start), 0, start, [start])]
    return astar_recursive(heap, {start: 0}, goal, graph, heuristic)

# ── 5C. A* on 2D Grid with Manhattan Distance ────────────────────────────────
def astar_grid(grid, start, end):
    """
    A* on a 2D grid — used in game engines and robotics.
    Uses Manhattan distance as heuristic.
    grid: 2D list  (0 = walkable, 1 = wall)
    """
    rows, cols = len(grid), len(grid[0])
    def h(pos): return abs(pos[0]-end[0]) + abs(pos[1]-end[1])
    heap = [(h(start), 0, start, [start])]
    g_costs = {start: 0}
    while heap:
        _, g, pos, path = heapq.heappop(heap)
        if pos == end:
            return path
        if g > g_costs.get(pos, float('inf')):
            continue
        r, c = pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                new_g = g + 1
                npos = (nr, nc)
                if new_g < g_costs.get(npos, float('inf')):
                    g_costs[npos] = new_g
                    heapq.heappush(heap, (new_g + h(npos), new_g, npos, path + [npos]))
    return None

# ── 5D. Real-World: 8-Puzzle Solver ──────────────────────────────────────────
GOAL_PUZZLE = (1,2,3,4,5,6,7,8,0)

def puzzle_heuristic(state):
    """Manhattan distance heuristic for 8-puzzle."""
    dist = 0
    for i, tile in enumerate(state):
        if tile != 0:
            goal_pos = GOAL_PUZZLE.index(tile)
            dist += abs(i//3 - goal_pos//3) + abs(i%3 - goal_pos%3)
    return dist

def get_puzzle_neighbors(state):
    neighbors = []
    idx = state.index(0)
    r, c = divmod(idx, 3)
    for move, (dr, dc) in [('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = list(state)
            new_idx = nr*3+nc
            new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[idx]
            neighbors.append((tuple(new_state), move, 1))
    return neighbors

def solve_8puzzle_astar(start):
    """
    Solve the 8-puzzle (sliding tile) using A*.
    Used in AI coursework and robotics motion planning.
    """
    heap = [(puzzle_heuristic(start), 0, start, [])]
    g_costs = {start: 0}
    while heap:
        _, g, state, moves = heapq.heappop(heap)
        if state == GOAL_PUZZLE:
            return moves
        if g > g_costs.get(state, float('inf')):
            continue
        for next_state, move, cost in get_puzzle_neighbors(state):
            new_g = g + cost
            if new_g < g_costs.get(next_state, float('inf')):
                g_costs[next_state] = new_g
                f = new_g + puzzle_heuristic(next_state)
                heapq.heappush(heap, (f, new_g, next_state, moves + [move]))
    return None


# =============================================================================
# 6.  GENETIC ALGORITHM (GA)
# =============================================================================

# ── 6A. GA Core Framework ────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    General-purpose Genetic Algorithm framework.
    Supports: encoding, selection, crossover, mutation, evolution.
    """
    def __init__(self, pop_size=100, mutation_rate=0.01,
                 crossover_rate=0.8, generations=500, elitism=2):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.elitism = elitism

    def initialize(self, gene_pool, chrom_length):
        return [[random.choice(gene_pool) for _ in range(chrom_length)]
                for _ in range(self.pop_size)]

    def tournament_selection(self, population, fitness_fn, k=3):
        """Select best individual from a random tournament of k candidates."""
        candidates = random.sample(population, k)
        return max(candidates, key=fitness_fn)

    def single_point_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1)-1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1[:], parent2[:]

    def two_point_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            p1, p2 = sorted(random.sample(range(1, len(parent1)), 2))
            c1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
            c2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
            return c1, c2
        return parent1[:], parent2[:]

    def mutate(self, chromosome, gene_pool):
        return [random.choice(gene_pool) if random.random() < self.mutation_rate else g
                for g in chromosome]

    def evolve(self, population, fitness_fn, gene_pool):
        """Run one generation of evolution."""
        sorted_pop = sorted(population, key=fitness_fn, reverse=True)
        new_pop = sorted_pop[:self.elitism]   # Keep elites
        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection(population, fitness_fn)
            p2 = self.tournament_selection(population, fitness_fn)
            c1, c2 = self.single_point_crossover(p1, p2)
            new_pop.append(self.mutate(c1, gene_pool))
            if len(new_pop) < self.pop_size:
                new_pop.append(self.mutate(c2, gene_pool))
        return new_pop

    def run(self, fitness_fn, gene_pool, chrom_length, target_fitness=None):
        """Full GA run. Returns (best_chromosome, best_fitness, history)."""
        population = self.initialize(gene_pool, chrom_length)
        history = []
        best_overall = None
        best_fitness = float('-inf')
        for gen in range(self.generations):
            population = self.evolve(population, fitness_fn, gene_pool)
            best = max(population, key=fitness_fn)
            fit = fitness_fn(best)
            history.append(fit)
            if fit > best_fitness:
                best_fitness = fit
                best_overall = best[:]
            if target_fitness and fit >= target_fitness:
                print(f"  Target reached at generation {gen+1}")
                break
        return best_overall, best_fitness, history


# ── 6B. GA Recursive (Evolution Loop as Recursion) ───────────────────────────
def ga_recursive(population, fitness_fn, gene_pool, ga, generations_left, best=None):
    """Genetic algorithm run recursively (each call = one generation)."""
    if generations_left == 0:
        return best or max(population, key=fitness_fn)
    population = ga.evolve(population, fitness_fn, gene_pool)
    current_best = max(population, key=fitness_fn)
    if best is None or fitness_fn(current_best) > fitness_fn(best):
        best = current_best[:]
    return ga_recursive(population, fitness_fn, gene_pool, ga, generations_left-1, best)


# ── 6C. Real-World: Traveling Salesman Problem (TSP) ─────────────────────────
def solve_tsp_ga(cities, pop_size=200, generations=1000):
    """
    Solve the Traveling Salesman Problem using GA.
    Finds an approximate shortest route visiting all cities.
    cities: list of (x, y) coordinates
    """
    n = len(cities)

    def route_distance(route):
        total = 0
        for i in range(n):
            a, b = cities[route[i]], cities[route[(i+1) % n]]
            total += math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return total

    def fitness(route): return 1 / route_distance(route)

    def ordered_crossover(p1, p2):
        """Order crossover (OX) for permutation chromosomes."""
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = p1[start:end]
        remaining = [g for g in p2 if g not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child

    def mutate_route(route):
        if random.random() < 0.02:
            i, j = random.sample(range(n), 2)
            route[i], route[j] = route[j], route[i]
        return route

    # Initialize population as shuffled routes
    population = [random.sample(range(n), n) for _ in range(pop_size)]
    best_route = None
    best_dist = float('inf')

    for gen in range(generations):
        sorted_pop = sorted(population, key=fitness, reverse=True)
        new_pop = sorted_pop[:2]  # Elitism
        while len(new_pop) < pop_size:
            p1 = max(random.sample(population, 3), key=fitness)
            p2 = max(random.sample(population, 3), key=fitness)
            child = ordered_crossover(p1, p2)
            new_pop.append(mutate_route(child))
        population = new_pop
        best = min(population, key=route_distance)
        dist = route_distance(best)
        if dist < best_dist:
            best_dist = dist
            best_route = best[:]

    return best_route, best_dist


# ── 6D. Real-World: Knapsack Problem ─────────────────────────────────────────
def solve_knapsack_ga(items, capacity, pop_size=100, generations=500):
    """
    Solve 0/1 Knapsack Problem using GA.
    items: list of (name, weight, value)
    Returns best selection of items within weight capacity.
    """
    n = len(items)

    def fitness(chromosome):
        total_weight = sum(items[i][1] for i in range(n) if chromosome[i] == 1)
        total_value  = sum(items[i][2] for i in range(n) if chromosome[i] == 1)
        if total_weight > capacity:
            return 0   # Penalty for exceeding capacity
        return total_value

    ga = GeneticAlgorithm(pop_size=pop_size, mutation_rate=0.02,
                          generations=generations, elitism=2)
    best, best_val, _ = ga.run(fitness, [0, 1], n)

    selected = [items[i][0] for i in range(n) if best[i] == 1]
    total_weight = sum(items[i][1] for i in range(n) if best[i] == 1)
    return selected, total_weight, best_val


# ── 6E. Real-World: Phrase Evolution (Hello World GA) ────────────────────────
def evolve_phrase_ga(target, pop_size=200, mutation_rate=0.01, generations=2000):
    """
    Evolve a random string of characters toward a target phrase.
    Classic GA demonstration. Shows convergence in action.
    """
    import string
    gene_pool = list(string.ascii_letters + string.digits + " .,!?'-")
    target_list = list(target)
    length = len(target)

    def fitness(chrom):
        return sum(1 for a, b in zip(chrom, target_list) if a == b)

    ga = GeneticAlgorithm(pop_size=pop_size, mutation_rate=mutation_rate,
                          generations=generations, elitism=5)
    best, best_fit, history = ga.run(fitness, gene_pool, length,
                                     target_fitness=length)
    return "".join(best), best_fit, history


# =============================================================================
# DEMO — Run all algorithms with example data
# =============================================================================
if __name__ == "__main__":

    print("=" * 65)
    print("  AI ALGORITHMS DEMO")
    print("=" * 65)

    # ── BFS ──────────────────────────────────────────────────────────
    print("\n[1] BFS — Basic Graph Traversal")
    graph = {'A':['B','C'],'B':['D','E'],'C':['F','G'],'D':[],'E':['H'],'F':[],'G':[],'H':[]}
    print("  Iterative:", bfs_iterative(graph, 'A', 'H'))
    print("  Recursive:", bfs_recursive_start(graph, 'A', 'H'))

    print("\n[1C] BFS — Social Network Degrees of Separation")
    network = {
        'Alice':  ['Bob', 'Carol'],
        'Bob':    ['Alice', 'Dave'],
        'Carol':  ['Alice', 'Eve'],
        'Dave':   ['Bob', 'Frank'],
        'Eve':    ['Carol'],
        'Frank':  ['Dave', 'Grace'],
        'Grace':  ['Frank'],
    }
    path, degree = degrees_of_separation(network, 'Alice', 'Grace')
    print(f"  Alice -> Grace: {' -> '.join(path)}  ({degree} degrees)")

    print("\n[1D] BFS — Web Crawler")
    link_map = {
        'home': ['about', 'products', 'blog'],
        'about': ['team', 'mission'],
        'products': ['item1', 'item2'],
        'blog': ['post1', 'post2'],
        'team': [], 'mission': [], 'item1': [], 'item2': [],
        'post1': ['post2'], 'post2': [],
    }
    crawled = web_crawler_bfs(link_map, 'home', max_pages=6)
    print("  Pages crawled:", crawled)

    # ── DFS ──────────────────────────────────────────────────────────
    print("\n[2] DFS — Basic Graph Traversal")
    print("  Iterative:", dfs_iterative(graph, 'A', 'H'))
    print("  Recursive:", dfs_recursive(graph, 'A', 'H'))

    print("\n[2C] DFS — Maze Solver")
    maze = [
        [0,0,1,0,0],
        [0,1,1,0,1],
        [0,0,0,0,0],
        [1,1,0,1,0],
        [0,0,0,1,0],
    ]
    path = solve_maze_dfs(maze, (0,0), (4,4))
    print("  Path through maze:", path)

    print("\n[2D] DFS — Dependency Resolver")
    deps = {
        'app':    ['flask', 'sqlalchemy'],
        'flask':  ['werkzeug', 'jinja2'],
        'sqlalchemy': ['greenlet'],
        'werkzeug': [], 'jinja2': ['markupsafe'],
        'markupsafe': [], 'greenlet': [],
    }
    order = topological_sort_dfs(deps)
    print("  Install order:", order)

    # ── UCS ──────────────────────────────────────────────────────────
    print("\n[3] UCS — Weighted Graph")
    wgraph = {
        'A': [('B',4),('C',1)],
        'C': [('B',2),('D',5)],
        'B': [('D',1)],
        'D': []
    }
    cost, path = ucs_iterative(wgraph, 'A', 'D')
    print(f"  Iterative: path={path}  cost={cost}")
    cost2, path2 = ucs_recursive_start(wgraph, 'A', 'D')
    print(f"  Recursive: path={path2}  cost={cost2}")

    print("\n[3C] UCS — Cheapest Flight Route")
    flights = {
        'Cairo':   [('Dubai',300), ('London',500)],
        'Dubai':   [('London',250), ('NYC',600)],
        'London':  [('NYC',400)],
        'NYC':     []
    }
    print(" ", cheapest_flight(flights, 'Cairo', 'NYC'))

    # ── GREEDY ───────────────────────────────────────────────────────
    print("\n[4] Greedy Best-First Search")
    ggraph = {'A':['B','C'],'B':['D','E'],'C':['F'],'D':[],'E':['G'],'F':['G'],'G':[]}
    heuristic = {'A':6,'B':4,'C':3,'D':5,'E':1,'F':2,'G':0}
    print("  Iterative:", greedy_iterative(ggraph, 'A', 'G', heuristic))
    print("  Recursive:", greedy_recursive_start(ggraph, 'A', 'G', heuristic))

    # ── A* ───────────────────────────────────────────────────────────
    print("\n[5] A* — Weighted Graph")
    agraph = {
        'S': [('A',1),('B',4)],
        'A': [('B',2),('C',5)],
        'B': [('C',1)],
        'C': []
    }
    ah = {'S':5,'A':3,'B':2,'C':0}
    c1, p1 = astar_iterative(agraph, 'S', 'C', ah)
    print(f"  Iterative: path={p1}  cost={c1}")
    c2, p2 = astar_recursive_start(agraph, 'S', 'C', ah)
    print(f"  Recursive: path={p2}  cost={c2}")

    print("\n[5C] A* — Grid Pathfinding")
    grid = [
        [0,0,1,0,0],
        [0,1,1,0,1],
        [0,0,0,0,0],
        [1,1,0,1,0],
        [0,0,0,1,0],
    ]
    gpath = astar_grid(grid, (0,0), (4,4))
    print("  Grid path:", gpath)

    print("\n[5D] A* — 8-Puzzle Solver")
    puzzle_start = (1,2,3,4,0,5,7,8,6)
    moves = solve_8puzzle_astar(puzzle_start)
    print(f"  Solved in {len(moves)} moves:", moves)

    # ── GENETIC ──────────────────────────────────────────────────────
    print("\n[6] Genetic Algorithm — Phrase Evolution")
    target = "Hello AI"
    best_phrase, fitness, history = evolve_phrase_ga(target, generations=3000)
    print(f"  Target : '{target}'")
    print(f"  Result : '{best_phrase}'  (fitness {fitness}/{len(target)})")

    print("\n[6C] GA — Knapsack Problem")
    items = [
        ('Laptop', 3, 4000),
        ('Phone',  1, 2500),
        ('Camera', 2, 3000),
        ('Book',   1,  500),
        ('Tablet', 2, 2000),
        ('Watch',  1, 1500),
    ]
    selected, weight, value = solve_knapsack_ga(items, capacity=5, generations=300)
    print(f"  Selected: {selected}")
    print(f"  Weight: {weight} kg   Value: ${value}")

    print("\n[6D] GA — TSP (Traveling Salesman)")
    cities = [(0,0),(1,3),(3,1),(5,4),(2,5),(4,2)]
    best_route, dist = solve_tsp_ga(cities, pop_size=100, generations=500)
    print(f"  Best route: {best_route}")
    print(f"  Total distance: {dist:.2f}")

    print("\n" + "=" * 65)
    print("  All algorithms completed successfully!")
    print("=" * 65)

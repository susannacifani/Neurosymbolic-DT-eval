import heapq
from itertools import product

# Env
pickaxe = (1, 1)
lava = (3, 3)
door = (3, 0)
gem = (0, 3)
grid_size = (4, 4)
obstacles = [] 
k = 84

empty_cells = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (3, 1),
                (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)]


def get_k_shortest_paths(start, goal, k=k, grid_size=(4, 4), obstacles=None):
    if obstacles is None: obstacles = []
    
    pq = [(0, [], start, {start})]
    found_paths = []
    
    while pq and len(found_paths) < k:
        dist, actions, (curr_x, curr_y), visited = heapq.heappop(pq)
        
        if (curr_x, curr_y) == goal:
            found_paths.append(actions)
            continue
            
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i, (dx, dy) in enumerate(moves):
            nx, ny = curr_x + dx, curr_y + dy

            if 0 <= nx < grid_size[1] and 0 <= ny < grid_size[0] and (nx, ny) not in obstacles:
                if (nx, ny) not in visited:
                    new_actions = actions + [i]
                    new_visited = visited | {(nx, ny)} 
                    heapq.heappush(pq, (len(new_actions), new_actions, (nx, ny), new_visited))
                    
    return found_paths




def task1():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Task 1.1: Start -> Pickaxe -> Lava
        paths1 = get_k_shortest_paths(start, pickaxe, obstacles=[(3, 3)])
        paths2 = get_k_shortest_paths(pickaxe, lava)
        for a, b in product(paths1, paths2):
            cell_paths.append(a + b)
        
        # Task 1.2: Start -> Lava -> Pickaxe
        paths3 = get_k_shortest_paths(start, lava, obstacles=[(1, 1)])
        paths4 = get_k_shortest_paths(lava, pickaxe)
        for a, b in product(paths3, paths4):
            cell_paths.append(a + b)
    
        cell_paths.sort(key=len)
        
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = 0
    for cella, percorsi in trajectories_by_cell.items():
        totale_diz = totale_diz + len(percorsi)

    return trajectories_by_cell, totale_diz


def task2():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava -> Door
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door])
        p2 = get_k_shortest_paths(pickaxe, lava, obstacles=[door])
        p3 = get_k_shortest_paths(lava, door)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)
            
        # Start -> Pickaxe -> Door -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door])
        p2 = get_k_shortest_paths(pickaxe, door, obstacles=[lava])
        p3 = get_k_shortest_paths(door, lava)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)

        # Start -> Lava -> Pickaxe -> Door
        p1 = get_k_shortest_paths(start, lava, obstacles=[pickaxe, door])
        p2 = get_k_shortest_paths(lava, pickaxe, obstacles=[door])
        p3 = get_k_shortest_paths(pickaxe, door)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)

        # Start -> Lava -> Door -> Pickaxe
        p1 = get_k_shortest_paths(start, lava, obstacles=[pickaxe, door])
        p2 = get_k_shortest_paths(lava, door, obstacles=[pickaxe])
        p3 = get_k_shortest_paths(door, pickaxe)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)

        # Start -> Door -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, door, obstacles=[pickaxe, lava])
        p2 = get_k_shortest_paths(door, pickaxe, obstacles=[lava])
        p3 = get_k_shortest_paths(pickaxe, lava)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)

        # Start -> Door -> Lava -> Pickaxe
        p1 = get_k_shortest_paths(start, door, obstacles=[pickaxe, lava])
        p2 = get_k_shortest_paths(door, lava, obstacles=[pickaxe])
        p3 = get_k_shortest_paths(lava, pickaxe)
        for a, b, c in product(p1, p2, p3):
            cell_paths.append(a + b + c)
    
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz


def task3():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava])
        p2 = get_k_shortest_paths(pickaxe, lava)
        for a, b in product(p1, p2):
            cell_paths.append(a + b)
            
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz



def task4():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []

        combinations = [
            [(start, pickaxe, [lava, door, gem]), (pickaxe, lava, [door, gem]), (lava, door, [gem]), (door, gem, [])],
            [(start, pickaxe, [lava, door, gem]), (pickaxe, door, [lava, gem]), (door, lava, [gem]), (lava, gem, [])],
            [(start, pickaxe, [lava, door, gem]), (pickaxe, door, [lava, gem]), (door, gem, [lava]), (gem, lava, [])],
            [(start, door, [pickaxe, lava, gem]), (door, pickaxe, [lava, gem]), (pickaxe, gem, [lava]), (gem, lava, [])],
            [(start, door, [pickaxe, lava, gem]), (door, pickaxe, [lava, gem]), (pickaxe, lava, [gem]), (lava, gem, [])],
            [(start, door, [pickaxe, lava, gem]), (door, gem, [pickaxe, lava]), (gem, pickaxe, [lava]), (pickaxe, lava, [])]
        ]
        
        for combo in combinations:
            k_interno = 4 
            p1 = get_k_shortest_paths(combo[0][0], combo[0][1], k=k_interno, obstacles=combo[0][2])
            p2 = get_k_shortest_paths(combo[1][0], combo[1][1], k=k_interno, obstacles=combo[1][2])
            p3 = get_k_shortest_paths(combo[2][0], combo[2][1], k=k_interno, obstacles=combo[2][2])
            p4 = get_k_shortest_paths(combo[3][0], combo[3][1], k=k_interno, obstacles=combo[3][2])
            
            for a, b, c, d in product(p1, p2, p3, p4):
                cell_paths.append(a + b + c + d)

        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz



def task5():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        k_interno = 5 
        
        # Start -> Door -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, door, k=k_interno, obstacles=[pickaxe, lava])
        p2 = get_k_shortest_paths(door, pickaxe, k=k_interno, obstacles=[lava])
        p3 = get_k_shortest_paths(pickaxe, lava, k=k_interno, obstacles=[])
        for a, b, c in product(p1, p2, p3): 
            cell_paths.append(a + b + c)

        # Start -> Pickaxe -> Door -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, k=k_interno, obstacles=[door, lava])
        p2 = get_k_shortest_paths(pickaxe, door, k=k_interno, obstacles=[lava])
        p3 = get_k_shortest_paths(door, lava, k=k_interno, obstacles=[])
        for a, b, c in product(p1, p2, p3): 
            cell_paths.append(a + b + c)

        # Start -> Pickaxe -> Lava -> Door
        p1 = get_k_shortest_paths(start, pickaxe, k=k_interno, obstacles=[door, lava])
        p2 = get_k_shortest_paths(pickaxe, lava, k=k_interno, obstacles=[door])
        p3 = get_k_shortest_paths(lava, door, k=k_interno, obstacles=[])
        for a, b, c in product(p1, p2, p3): 
            cell_paths.append(a + b + c)

        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz



def task6():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    valid_orders = [
        [pickaxe, lava, door, gem],
        [pickaxe, lava, gem, door],
        [pickaxe, door, lava, gem],
        [pickaxe, door, gem, lava],
        [pickaxe, gem, lava, door],
        [pickaxe, gem, door, lava],
        [door, pickaxe, lava, gem],
        [door, pickaxe, gem, lava],
        [door, gem, pickaxe, lava],
        [gem, pickaxe, lava, door],
        [gem, pickaxe, door, lava],
        [gem, door, pickaxe, lava]
    ]

    for start in empty_cells:
        cell_paths = []
        k_interno = 3 
        
        for order in valid_orders:
            p1 = get_k_shortest_paths(start, order[0], k=k_interno, obstacles=order[1:])
            p2 = get_k_shortest_paths(order[0], order[1], k=k_interno, obstacles=order[2:])
            p3 = get_k_shortest_paths(order[1], order[2], k=k_interno, obstacles=order[3:])
            p4 = get_k_shortest_paths(order[2], order[3], k=k_interno, obstacles=[])
            
            for a, b, c, d in product(p1, p2, p3, p4):
                cell_paths.append(a + b + c + d)

        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz


def task7():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door])
        p2 = get_k_shortest_paths(pickaxe, lava, obstacles=[door])
        
        for a, b in product(p1, p2):
            cell_paths.append(a + b)
        
        # Start -> Lava -> Pickaxe
        p3 = get_k_shortest_paths(start, lava, obstacles=[pickaxe, door])
        p4 = get_k_shortest_paths(lava, pickaxe, obstacles=[door])
        
        for a, b in product(p3, p4):
            cell_paths.append(a + b)
    
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz


def task8():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door, gem])
        p2 = get_k_shortest_paths(pickaxe, lava, obstacles=[door, gem])
        
        for a, b in product(p1, p2):
            cell_paths.append(a + b)
        
        # Start -> Lava -> Pickaxe
        p3 = get_k_shortest_paths(start, lava, obstacles=[pickaxe, door, gem])
        p4 = get_k_shortest_paths(lava, pickaxe, obstacles=[door, gem])
        
        for a, b in product(p3, p4):
            cell_paths.append(a + b)
    
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz


def task9():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door])
        p2 = get_k_shortest_paths(pickaxe, lava, obstacles=[door])
        
        for a, b in product(p1, p2):
            cell_paths.append(a + b)
            
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz



def task10():
    trajectories_by_cell = {cell: [] for cell in empty_cells}

    for start in empty_cells:
        cell_paths = []
        
        # Start -> Pickaxe -> Lava
        p1 = get_k_shortest_paths(start, pickaxe, obstacles=[lava, door, gem])
        p2 = get_k_shortest_paths(pickaxe, lava, obstacles=[door, gem])
        
        for a, b in product(p1, p2):
            cell_paths.append(a + b)
            
        cell_paths.sort(key=len)
        trajectories_by_cell[start] = cell_paths[:k]

    totale_diz = sum(len(percorsi) for percorsi in trajectories_by_cell.values())
    return trajectories_by_cell, totale_diz




# if __name__ == "__main__":
#     diz, tot = task1()
#     print(diz)
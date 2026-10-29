"""Percorsi ottimi per una disposizione qualsiasi della griglia.

Ricerca in ampiezza (BFS) sugli stati (casella, stato dell'automa): ogni mossa porta in una nuova
casella e l'automa avanza con il simbolo di quella casella. Gli stati di fallimento (reward -100,
es. toccare un oggetto da evitare) non si possono attraversare; ci si ferma al primo stato
accettante. La BFS trova sempre un percorso di lunghezza minima, per qualunque formula e
disposizione. Tra i percorsi minimi ne sceglie uno a caso (rng), cosi' i dataset "expert" non
contengono sempre lo stesso percorso.

Sostituisce DijkstraTraj.py della griglia 4x4, che conteneva percorsi scritti a mano per le
posizioni fisse degli oggetti.
"""
from collections import deque

import numpy as np

ACTIONS = (0, 1, 2, 3)


def _neighbors(env, pos, q):
    """(azione, nuova casella, nuovo stato dell'automa) per ogni mossa da (pos, q)"""
    for a in ACTIONS:
        nxt = np.clip(pos + env._action_to_direction[a], 0, env.size - 1)
        q_next = env.automaton.transitions[q][int(env.grid[nxt[0], nxt[1]])]
        yield a, (int(nxt[0]), int(nxt[1])), q_next


def shortest_path(env, rng=None):
    """Azioni di un percorso minimo dalla situazione attuale dell'ambiente (dopo reset) fino alla
    vittoria, evitando la sconfitta; None se la vittoria non e' raggiungibile."""
    aut = env.automaton
    start = (int(env._agent_location[0]), int(env._agent_location[1]), env.curr_automaton_state)
    if aut.acceptance[start[2]]:
        return []

    # BFS: distanza dall'inizio e tutti i predecessori che stanno su un percorso minimo
    dist = {start: 0}
    preds = {start: []}
    frontier = deque([start])
    goals = []
    while frontier and not goals:
        layer_size = len(frontier)
        for _ in range(layer_size):
            x, y, q = frontier.popleft()
            for a, (nx, ny), q_next in _neighbors(env, np.array([x, y]), q):
                if aut.rewards[q_next] == -100:
                    continue   # stato di fallimento: vietato
                node = (nx, ny, q_next)
                if node not in dist:
                    dist[node] = dist[(x, y, q)] + 1
                    preds[node] = []
                    frontier.append(node)
                    if aut.acceptance[q_next]:
                        goals.append(node)
                if dist[node] == dist[(x, y, q)] + 1:
                    preds[node].append(((x, y, q), a))
    if not goals:
        return None

    # ricostruzione all'indietro scegliendo a caso tra i predecessori ottimi
    rng = rng if rng is not None else np.random.default_rng()
    node = goals[rng.integers(len(goals))]
    actions = []
    while node != start:
        prev, a = preds[node][rng.integers(len(preds[node]))]
        actions.append(a)
        node = prev
    return actions[::-1]

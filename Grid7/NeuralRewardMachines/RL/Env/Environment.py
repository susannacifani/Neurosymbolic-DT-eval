"""Grid world 7x7 con oggetti in posizioni casuali a ogni episodio.

Differenze rispetto alla griglia 4x4 (Grid4):
- griglia SIZE x SIZE (default 7) con COPIES copie di ogni oggetto (default 2: 8 oggetti);
- a ogni reset la disposizione degli oggetti e la partenza dell'agente sono casuali; con lo
  stesso seed si ottiene sempre lo stesso episodio;
- l'osservazione contiene anche la disposizione degli oggetti, che non e' piu' fissa:
    symbolic: mappa (5, SIZE, SIZE) con un canale per tipo di oggetto (P, L, D, G) e uno per l'agente
    image:    immagine (3, H, W) in [0, 1] della griglia, nello stile della 4x4 (robot a grandezza
              quasi piena al centro della sua casella)
Formule, automa e reward sono identici: il simbolo e' l'oggetto nella casella dell'agente.
"""
import os
import sys

import gym
import numpy as np
from gym import spaces
from PIL import Image, ImageDraw

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base)

from RL.Env.FiniteStateMachine import MooreMachine

REWARD_SCHEMES = {"distance": "distance", "three_value": "three_value_acceptance"}

# simboli: indici usati dall'automa (come nella 4x4)
SYMBOLS = ['P', 'L', 'D', 'G', 'E']
PICKAXE, LAVA, DOOR, GEM, EMPTY = range(5)
OBJECT_TYPES = [PICKAXE, LAVA, DOOR, GEM]

# azioni (come nella 4x4): 0 = y+1, 1 = x+1, 2 = y-1, 3 = x-1
ACTION_TO_DIRECTION = {0: np.array([0, 1]), 1: np.array([1, 0]), 2: np.array([0, -1]), 3: np.array([-1, 0])}

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")
ICON_FILES = {PICKAXE: "pickaxe.png", LAVA: "lava.jpg", DOOR: "door.png", GEM: "gem.png"}
ROBOT_FILE = "robot.png"


def sample_layout(rng, size=7, copies=2):
    """Disposizione casuale: griglia (size, size) con il simbolo di ogni casella (EMPTY se vuota).
    COPIES copie per ogni tipo di oggetto, tutte in caselle diverse."""
    grid = np.full((size, size), EMPTY, dtype=np.int64)
    cells = rng.choice(size * size, size=copies * len(OBJECT_TYPES), replace=False)
    for i, cell in enumerate(cells):
        grid[cell // size, cell % size] = OBJECT_TYPES[i // copies]
    return grid


def symbolic_observation(grid, agent_pos):
    """Mappa (5, size, size): canali 0-3 = posizioni di P, L, D, G; canale 4 = agente.
    grid e' indicizzata [x, y] come le posizioni."""
    size = grid.shape[0]
    obs = np.zeros((len(OBJECT_TYPES) + 1, size, size), dtype=np.float32)
    for c, obj in enumerate(OBJECT_TYPES):
        obs[c][grid == obj] = 1.0
    obs[len(OBJECT_TYPES), agent_pos[0], agent_pos[1]] = 1.0
    return obs


class ImageRenderer:
    """Disegna la griglia nello stile della 4x4 (sfondo bianco, linee nere, le stesse icone PNG),
    componendola "a tasselli": ogni casella puo' contenere solo 5 cose (P, L, D, G, vuota), con o
    senza il robot sopra, quindi esistono solo 10 tasselli. Ognuno si disegna una volta sola ad alta
    risoluzione (HI_CELL pixel, come i 128 pixel per casella della 4x4) e si rimpicciolisce a
    CELL_PX pixel con un filtro che lo ammorbidisce; un'immagine e' poi solo la griglia dei tasselli.
    Cosi' disegnare un'immagine e' istantaneo (e si puo' fare anche sulla GPU dai codici delle
    caselle, vedi cell_codes), ed e' sempre identica.
    Il robot e' al centro della casella, sopra l'oggetto se c'e', un po' piu' piccolo della casella
    (ROBOT_SCALE): a grandezza piena coprirebbe quasi del tutto le icone sottili (pickaxe, gem).
    Uscita (3, size * CELL_PX, size * CELL_PX) in [0, 1]."""

    HI_CELL = 128          # pixel per casella nel disegno ad alta risoluzione (come la 4x4)
    LINE = 2               # spessore del bordo di ogni tassello (due caselle vicine: 2 + 2)
    ROBOT_SCALE = 0.7      # grandezza del robot rispetto alla casella
    NUM_CONTENTS = 5       # P, L, D, G, vuota

    def __init__(self, size=7, out_px=112):
        if out_px % size:
            raise ValueError("out_px deve essere un multiplo di size")
        self.size, self.px, self.cell = size, out_px, out_px // size
        icon = self.HI_CELL - 2 * self.LINE
        icons = {obj: Image.open(os.path.join(IMG_DIR, name)).convert("RGBA").resize((icon, icon), Image.LANCZOS)
                 for obj, name in ICON_FILES.items()}
        robot_px = int(icon * self.ROBOT_SCALE)
        robot = Image.open(os.path.join(IMG_DIR, ROBOT_FILE)).convert("RGBA").resize((robot_px, robot_px), Image.LANCZOS)
        tiles = []
        for has_robot in (False, True):
            for content in range(self.NUM_CONTENTS):
                tile = Image.new("RGBA", (self.HI_CELL, self.HI_CELL), (255, 255, 255, 255))
                draw = ImageDraw.Draw(tile)
                draw.rectangle([0, 0, self.HI_CELL - 1, self.HI_CELL - 1], outline=(0, 0, 0, 255), width=self.LINE)
                if content != EMPTY:
                    tile.alpha_composite(icons[content], (self.LINE, self.LINE))
                if has_robot:
                    offset = (self.HI_CELL - robot_px) // 2
                    tile.alpha_composite(robot, (offset, offset))
                small = tile.convert("RGB").resize((self.cell, self.cell), Image.LANCZOS)
                tiles.append(np.transpose(np.asarray(small, dtype=np.float32) / 255.0, (2, 0, 1)))
        # tiles[codice]: codice = contenuto + NUM_CONTENTS * (robot presente)
        self.tiles = np.stack(tiles)

    def cell_codes(self, grid, agent_pos):
        """codice del tassello di ogni casella, indicizzato [x, y] come grid"""
        codes = np.array(grid, dtype=np.int64).copy()
        codes[int(agent_pos[0]), int(agent_pos[1])] += self.NUM_CONTENTS
        return codes

    def render_codes(self, codes):
        """immagine (3, H, W) da una griglia di codici (x, y)"""
        c, n = self.cell, self.size
        # tasselli in ordine (y, x): riga dell'immagine = y, colonna = x
        t = self.tiles[codes.T]                              # (n_y, n_x, 3, c, c)
        return t.transpose(2, 0, 3, 1, 4).reshape(3, n * c, n * c)

    def render(self, grid, agent_pos):
        return self.render_codes(self.cell_codes(grid, agent_pos))


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, formula, render_mode="rgb_array", state_type="symbolic", use_dfa_state=False,
                 train=False, size=7, reward_type=None, copies=2, max_num_steps=100, image_px=112):
        if reward_type not in REWARD_SCHEMES:
            raise ValueError(f"reward_type must be one of {list(REWARD_SCHEMES)}, got {reward_type!r}")
        if state_type not in ("symbolic", "image"):
            raise ValueError(f"state_type must be 'symbolic' or 'image', got {state_type!r}")
        self.reward_type = reward_type
        self.dictionary_symbols = list(SYMBOLS)
        self.state_type = state_type
        self.use_dfa_state = use_dfa_state
        self.size = size
        self.copies = copies
        self.max_num_steps = max_num_steps
        self.curr_step = 0
        self.render_mode = render_mode

        self.formula = formula
        self.task = formula[2]
        self.automaton = MooreMachine(formula[0], formula[1], formula[2], reward=REWARD_SCHEMES[reward_type],
                                      dictionary_symbols=self.dictionary_symbols)
        self.max_reward = 100
        self.list_rew = sorted(set(self.automaton.rewards))
        self.rew_dictionary = {reward: idx for idx, reward in enumerate(self.list_rew)}

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = ACTION_TO_DIRECTION
        self.renderer = ImageRenderer(size, image_px) if state_type == "image" else None
        if state_type == "symbolic":
            self.state_space_size = (len(OBJECT_TYPES) + 1, size, size)
        else:
            self.state_space_size = (3, self.renderer.px, self.renderer.px)

        self.grid = None
        self._agent_location = None
        self.curr_automaton_state = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, grid=None, start_pos=None):
        """seed: disposizione e partenza casuali ma riproducibili; grid / start_pos: per fissarle
        esplicitamente (es. per ricostruire un episodio salvato)."""
        rng = np.random.default_rng(seed)
        self.grid = np.array(grid, dtype=np.int64) if grid is not None else sample_layout(rng, self.size, self.copies)
        if start_pos is None:
            empty = np.argwhere(self.grid == EMPTY)
            start_pos = empty[rng.integers(len(empty))]
        self._agent_location = np.array(start_pos, dtype=np.int64)
        self.curr_automaton_state = 0
        self.curr_step = 0
        return self._observation(), 0, self.rew_dictionary[0]

    def _current_symbol(self):
        return int(self.grid[self._agent_location[0], self._agent_location[1]])

    def step(self, action):
        self.curr_step += 1
        self._agent_location = np.clip(self._agent_location + ACTION_TO_DIRECTION[int(action)], 0, self.size - 1)

        new_state = self.automaton.transitions[self.curr_automaton_state][self._current_symbol()]
        if new_state == self.curr_automaton_state:
            reward = 0
        else:
            reward = self.automaton.rewards[new_state] - self.automaton.rewards[self.curr_automaton_state]
        potential = self.automaton.rewards[new_state]
        self.curr_automaton_state = new_state

        done = potential in (100, -100)       # vittoria o fallimento
        truncated = self.curr_step >= self.max_num_steps
        return self._observation(), reward, done, truncated, self.rew_dictionary[potential]

    def _observation(self):
        if self.state_type == "symbolic":
            obs = symbolic_observation(self.grid, self._agent_location)
        else:
            obs = self.renderer.render(self.grid, self._agent_location)
        if self.use_dfa_state:
            one_hot = np.zeros(self.automaton.num_of_states, dtype=np.float32)
            one_hot[self.curr_automaton_state] = 1.0
            return [one_hot, obs]
        return obs

    def render(self):
        """immagine (H, W, 3) uint8 della situazione attuale, per visualizzarla"""
        renderer = self.renderer or ImageRenderer(self.size)
        return (np.transpose(renderer.render(self.grid, self._agent_location), (1, 2, 0)) * 255).round().astype(np.uint8)

    def get_automaton_specs(self):
        num_of_states = self.automaton.num_of_states
        num_of_symbols = len(self.dictionary_symbols)
        num_outputs = len(self.list_rew)
        transition_function = self.automaton.transitions
        automaton_rewards = [self.rew_dictionary[rew] for rew in self.automaton.rewards]
        return num_of_states, num_of_symbols, num_outputs, transition_function, automaton_rewards

    def close(self):
        pass

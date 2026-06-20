"""
Main entry-point: trains both agents, then launches a pygame side-by-side
comparison showing the Baseline RL agent vs the MetaMo agent in real time.

Controls:
    SPACE  — pause / resume
    R      — reset both agents to a fresh episode
    Q/ESC  — quit
    F/UP   — faster  (increase speed)
    S/DOWN — slower  (decrease speed)
"""

import sys, os, math
import numpy as np
import pygame

#   path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
for path in (REPO_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from environment.gridworld   import (
    GridWorld,
    GRID_SIZE,
    LAVA_CELLS,
    MAX_STEPS,
    MINERAL_SPAWN_BAND,
)
from agents.baseline_agent   import BaselineAgent
from agents.metamo_agent     import MetaMoAgent
from metrics.collector       import MetricsCollector, EpisodeLog
from metamo.core             import (
    arousal as mot_arousal,
    in_safe_region as mot_in_safe_region,
    safety_threshold as mot_safety_threshold,
)
from core.config             import G_IND, G_TRANS

#  Training config

TRAIN_EPISODES = 100          
EVAL_EPISODES  = 100
MAX_STEPS_EP   = MAX_STEPS
DEFAULT_STEPS_PER_SECOND = 8.0
MIN_STEPS_PER_SECOND = 1.0
MAX_STEPS_PER_SECOND = 30.0
DANGER_DISTANCE = MINERAL_SPAWN_BAND

#  Pygame layout

CELL          = 48            
GRID_PX       = CELL * GRID_SIZE    
PANEL_W       = 320            
GAP           = 28            
DASH_GAP      = 16
MARGIN        = 20
GRID_OY       = 70
DASH_PANEL_H  = 335
WIN_W         = MARGIN * 2 + GRID_PX * 2 + GAP + DASH_GAP + PANEL_W
WIN_H         = GRID_OY + DASH_PANEL_H * 2 + DASH_GAP + 42
FPS           = 60

# Colours
BG            = (18,  20,  30)
GRID_LINE     = (40,  44,  60)
SAFE_CELL     = (28,  32,  46)
LAVA_COLOR    = (210,  60,  20)
MINERAL_COLOR = (80,  220, 140)
AGENT_BASE    = (100, 160, 255)
AGENT_META    = (255, 210,  60)
PANEL_BG      = (24,  26,  40)
TEXT_COLOR    = (220, 225, 240)
DIM_TEXT      = (120, 128, 160)
ACCENT_BL     = (80,  160, 255)
ACCENT_ME     = (255, 200,  50)
RED           = (220,  60,  60)
GREEN         = (80,  200, 120)
AMBER         = (255, 180,  60)
BAR_BG        = (50,  54,  72)

ASSET_DIR     = os.path.join(ROOT, "assets")
FASTER_KEYS   = {
    pygame.K_f,
    pygame.K_UP,
    pygame.K_EQUALS,
    getattr(pygame, "K_PLUS", pygame.K_EQUALS),
    getattr(pygame, "K_KP_PLUS", pygame.K_EQUALS),
}
SLOWER_KEYS   = {
    pygame.K_s,
    pygame.K_DOWN,
    pygame.K_MINUS,
    getattr(pygame, "K_KP_MINUS", pygame.K_MINUS),
}


def lava_cells_from_state(env_state: dict) -> tuple[tuple[int, int], ...]:
    return tuple(env_state.get("lava_cells", LAVA_CELLS))


def in_environment_unsafe_zone(env_state: dict) -> bool:
    return bool(env_state["in_lava"] or env_state["lava_distance"] <= DANGER_DISTANCE)


def environment_region(env_state: dict) -> tuple[str, tuple[int, int, int]]:
    if env_state["in_lava"]:
        return "LAVA", RED
    if env_state["lava_distance"] <= DANGER_DISTANCE:
        return "DANGER", AMBER
    return "SAFE", GREEN


def mean_from_summary(summary: dict, key: str) -> float:
    if not summary:
        return 0.0
    return float(summary.get(key, {}).get("mean", 0.0))


#  Silent training loop

def train_agent(agent, label: str, episodes: int, seed_offset: int = 0):
    print(f"Training {label} for {episodes} episodes ...", end="", flush=True)
    for ep in range(episodes):
        env = GridWorld(seed=ep + seed_offset, max_steps=MAX_STEPS_EP)
        state = env.reset()
        agent.reset_episode()
        done = False
        for _ in range(MAX_STEPS_EP):
            if isinstance(agent, MetaMoAgent):
                action, alpha = agent.select_action(state)
            else:
                action = agent.select_action(state)
                alpha  = None

            next_state, reward, done, info = env.step(action)

            if isinstance(agent, MetaMoAgent):
                agent.update(state, action, reward, next_state, done,
                             info.get("event"), alpha)
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        agent.decay_epsilon()
        if (ep + 1) % 50 == 0:
            print(".", end="", flush=True)

    print(" done.")
    agent.epsilon = 0.12 if isinstance(agent, MetaMoAgent) else 0.05


#  Drawing helpers

def draw_grid(surf, offset_x, offset_y, env_state, agent_surf, mineral_surf,
              is_metamo: bool, mot_state=None, alpha_dict=None):
    """Draw one 10×10 grid with all decorations."""

    lava_cells = lava_cells_from_state(env_state)

    # Background cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rx = offset_x + c * CELL
            ry = offset_y + r * CELL
            rect = pygame.Rect(rx, ry, CELL, CELL)

            cell = (r, c)
            if cell in lava_cells:
                # Animated lava cell
                t = pygame.time.get_ticks() / 1000.0
                wave = int(10 * math.sin(t * 2 + c * 0.7 + r * 0.4))
                col = (
                    min(255, LAVA_COLOR[0] + wave),
                    max(0,   LAVA_COLOR[1] + wave // 2),
                    LAVA_COLOR[2],
                )
                pygame.draw.rect(surf, col, rect)
            else:
                pygame.draw.rect(surf, SAFE_CELL, rect)

            pygame.draw.rect(surf, GRID_LINE, rect, 1)

    # Mineral
    mr, mc = env_state["mineral_pos"]
    mx = offset_x + mc * CELL
    my = offset_y + mr * CELL
    # Pulse glow
    t  = pygame.time.get_ticks() / 1000.0
    glow_r = int(CELL * 0.4 + 4 * math.sin(t * 3))
    glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*MINERAL_COLOR, 60), (glow_r, glow_r), glow_r)
    surf.blit(glow_surf, (mx + CELL//2 - glow_r, my + CELL//2 - glow_r))
    if mineral_surf:
        ms = pygame.transform.scale(mineral_surf, (CELL - 6, CELL - 6))
        surf.blit(ms, (mx + 3, my + 3))
    else:
        pygame.draw.circle(surf, MINERAL_COLOR, (mx + CELL//2, my + CELL//2), CELL//3)

    # Agent
    ar, ac = env_state["pos"]
    ax = offset_x + ac * CELL
    ay = offset_y + ar * CELL
    color = AGENT_META if is_metamo else AGENT_BASE
    if agent_surf:
        ags = pygame.transform.scale(agent_surf, (CELL - 4, CELL - 4))
        # Tint the sprite
        tinted = ags.copy()
        tint = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
        tc = AGENT_META if is_metamo else AGENT_BASE
        tint.fill((*tc, 80))
        tinted.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(ags, (ax + 2, ay + 2))
    else:
        pygame.draw.circle(surf, color, (ax + CELL//2, ay + CELL//2), CELL//2 - 4)

    # Border
    border_col = ACCENT_ME if is_metamo else ACCENT_BL
    pygame.draw.rect(surf, border_col,
                     pygame.Rect(offset_x - 2, offset_y - 2,
                                 GRID_PX + 4, GRID_PX + 4), 2)


def draw_bar(surf, x, y, w, value, color, label, font_xs):
    value = max(0.0, min(1.0, float(value)))
    txt = font_xs.render(f"{label} {value:.2f}", True, TEXT_COLOR)
    surf.blit(txt, (x, y))
    by = y + 12
    pygame.draw.rect(surf, BAR_BG, (x, by, w, 7))
    filled = int(w * value)
    if filled > 0:
        pygame.draw.rect(surf, color, (x, by, filled, 7))
    pygame.draw.rect(surf, DIM_TEXT, (x, by, w, 7), 1)


def draw_metric_line(surf, x, y, w, label, value, font_xs, value_color=TEXT_COLOR):
    lab = font_xs.render(label, True, DIM_TEXT)
    val = font_xs.render(value, True, value_color)
    surf.blit(lab, (x, y))
    surf.blit(val, (x + w - val.get_width(), y))


def draw_panel(surf, ox, oy, panel_h, label, color,
               env_state, episode, completed_episodes, total_reward,
               ep_log, summary,
               font_title, font_xs,
               mot_state=None, alpha_dict=None):
    """Stacked dashboard panel for one agent."""
    pygame.draw.rect(surf, PANEL_BG, (ox, oy, PANEL_W, panel_h), border_radius=6)
    pygame.draw.rect(surf, color, (ox, oy, PANEL_W, 4), border_top_left_radius=6, border_top_right_radius=6)

    t = font_title.render(label, True, color)
    surf.blit(t, (ox + 12, oy + 10))

    live_steps = max(ep_log.total_steps, env_state["step"])
    live_lava_rate = ep_log.lava_steps / live_steps if live_steps else 0.0
    live_unsafe_rate = ep_log.unsafe_rate()
    live_srv_rate = ep_log.srv_rate()
    region_text, region_color = environment_region(env_state)

    avg_lava = mean_from_summary(summary, "lava_rate")
    avg_unsafe = mean_from_summary(summary, "unsafe_rate")
    avg_srv = mean_from_summary(summary, "srv_rate")

    x = ox + 12
    w = PANEL_W - 24
    y = oy + 40
    line_h = 16

    rows = [
        ("Episode", f"{episode}/{EVAL_EPISODES}", TEXT_COLOR),
        ("Completed", f"{completed_episodes}/{EVAL_EPISODES}", TEXT_COLOR),
        ("Step", f"{env_state['step']}/{MAX_STEPS_EP}", TEXT_COLOR),
        ("Minerals", f"{ep_log.minerals_collected}/{ep_log.minerals_spawned}", MINERAL_COLOR),
        ("Reward", f"{total_reward:+.0f}", TEXT_COLOR),
        ("Lava ep/avg", f"{live_lava_rate:.2f}/{avg_lava:.2f}", RED if live_lava_rate else TEXT_COLOR),
        ("Unsafe ep/avg", f"{live_unsafe_rate:.2f}/{avg_unsafe:.2f}", AMBER if live_unsafe_rate else TEXT_COLOR),
        ("Env region", region_text, region_color),
    ]

    for row_label, row_value, row_color in rows:
        draw_metric_line(surf, x, y, w, row_label, row_value, font_xs, row_color)
        y += line_h

    if mot_state is not None:
        safe = mot_in_safe_region(mot_state)
        safe_color = GREEN if safe else RED
        draw_metric_line(surf, x, y, w, "MetaMo SRV ep/avg", f"{live_srv_rate:.2f}/{avg_srv:.2f}", font_xs, safe_color)
        y += line_h
        draw_metric_line(surf, x, y, w, "Appraisal risk", f"{alpha_dict.get('risk', 0.0):.2f}", font_xs, RED if alpha_dict.get("risk", 0.0) > 0.5 else TEXT_COLOR)
        y += line_h + 4

        target_ind = alpha_dict.get("target_individuation", mot_state.G[G_IND])
        target_trans = alpha_dict.get("target_transcendence", mot_state.G[G_TRANS])
        draw_bar(surf, x, y, w, mot_state.G[G_IND], GREEN, "Individuation", font_xs); y += 22
        draw_bar(surf, x, y, w, target_ind, (120, 220, 150), "Consensus Ind", font_xs); y += 22
        draw_bar(surf, x, y, w, mot_state.G[G_TRANS], ACCENT_ME, "Transcendence", font_xs); y += 22
        draw_bar(surf, x, y, w, target_trans, (255, 225, 100), "Consensus Trans", font_xs); y += 22
        draw_bar(surf, x, y, w, mot_safety_threshold(mot_state), (60, 200, 120), "Safety threshold", font_xs); y += 22
        draw_bar(surf, x, y, w, mot_arousal(mot_state), (220, 80, 80), "Arousal", font_xs)
    else:
        draw_metric_line(surf, x, y, w, "Env SRV ep/avg", f"{live_srv_rate:.2f}/{avg_srv:.2f}", font_xs, AMBER if live_srv_rate else TEXT_COLOR)
        y += line_h + 6
        energy = env_state["energy"]
        ec = GREEN if energy > 50 else AMBER if energy > 25 else RED
        draw_bar(surf, x, y, w, energy / 100.0, ec, f"Energy {energy:.0f}/100", font_xs)



def main():
    # Train silently  
    baseline = BaselineAgent(seed=0)
    metamo   = MetaMoAgent(seed=0)
    train_agent(baseline, "Baseline RL", TRAIN_EPISODES, seed_offset=0)
    train_agent(metamo,   "MetaMo  RL",  TRAIN_EPISODES, seed_offset=0)

    # Metrics collectors  
    bl_metrics  = MetricsCollector("Baseline")
    mm_metrics  = MetricsCollector("MetaMo")

    # Pygame init
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        pass
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("MetaMo vs Baseline RL — GridWorld Evaluation")
    clock  = pygame.time.Clock()

    # Fonts
    font_title = pygame.font.Font(None, 28)
    font_xs    = pygame.font.Font(None, 17)
    font_lg    = pygame.font.Font(None, 40)

    # Assets
    try:
        potato_raw = pygame.image.load(os.path.join(ASSET_DIR, "cat.jpg")).convert_alpha()
        potato_img = potato_raw
    except Exception:
        potato_img = None

    mineral_surf = None   

    try:
        clank_sound = pygame.mixer.Sound(os.path.join(ASSET_DIR, "clank.wav"))
        clank_sound.set_volume(0.4)
    except Exception:
        clank_sound = None

    # Simulation state 
    def new_episode(ep_num):
        env_bl = GridWorld(seed=ep_num * 7, max_steps=MAX_STEPS_EP)
        env_mm = GridWorld(seed=ep_num * 7, max_steps=MAX_STEPS_EP)    # same seed → same env
        baseline.reset_episode()
        metamo.reset_episode()
        s_bl = env_bl.reset()
        s_mm = env_mm.reset()
        return env_bl, env_mm, s_bl, s_mm

    def clear_episode_logs():
        return False, False, 0.0, 0.0, 0, 0, EpisodeLog(), EpisodeLog()

    episode      = 1
    completed_episodes = 0
    env_bl, env_mm, s_bl, s_mm = new_episode(episode)

    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, ep_log_bl, ep_log_mm = clear_episode_logs()
    alpha_mm = {"risk":0,"urgency":0,"eu":0}

    steps_per_second = DEFAULT_STEPS_PER_SECOND
    step_accum_ms = 0.0
    paused     = False
    evaluation_complete = False

    running = True
    while running:
        dt_ms = clock.tick(FPS)

        # Events   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    if evaluation_complete:
                        bl_metrics.episodes.clear()
                        mm_metrics.episodes.clear()
                        completed_episodes = 0
                        episode = 1
                    evaluation_complete = False
                    paused = False
                    step_accum_ms = 0.0
                    env_bl, env_mm, s_bl, s_mm = new_episode(episode)
                    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, ep_log_bl, ep_log_mm = clear_episode_logs()
                if event.key in FASTER_KEYS:
                    steps_per_second = min(steps_per_second + 1.0, MAX_STEPS_PER_SECOND)
                if event.key in SLOWER_KEYS:
                    steps_per_second = max(steps_per_second - 1.0, MIN_STEPS_PER_SECOND)

        #  Simulation steps  
        if not paused and not evaluation_complete:
            step_accum_ms += dt_ms
            step_interval_ms = 1000.0 / steps_per_second
            while step_accum_ms >= step_interval_ms and not evaluation_complete:
                step_accum_ms -= step_interval_ms

                # Baseline step
                if not done_bl:
                    a_bl = baseline.select_action(s_bl)
                    ns_bl, r_bl, done_bl, info_bl = env_bl.step(a_bl)
                    baseline.update(s_bl, a_bl, r_bl, ns_bl, done_bl)
                    reward_bl += r_bl
                    if ns_bl["in_lava"]:
                        lava_bl += 1
                        if clank_sound: clank_sound.play()
                    if info_bl.get("event") == "mineral":
                        ep_log_bl.minerals_collected += 1
                    ep_log_bl.total_steps = env_bl.step_count
                    ep_log_bl.total_reward = reward_bl
                    ep_log_bl.lava_steps = lava_bl
                    ep_log_bl.minerals_spawned = env_bl.minerals_spawned
                    ep_log_bl.energy_log.append(ns_bl["energy"])
                    ep_log_bl.survived = ns_bl["energy"] > 0
                    bl_unsafe = in_environment_unsafe_zone(ns_bl)
                    ep_log_bl.unsafe_flags.append(bl_unsafe)
                    ep_log_bl.srv_flags.append(bl_unsafe)
                    s_bl = ns_bl

                # MetaMo step
                if not done_mm:
                    a_mm, alpha_mm = metamo.select_action(s_mm)
                    ns_mm, r_mm, done_mm, info_mm = env_mm.step(a_mm)
                    metamo.update(s_mm, a_mm, r_mm, ns_mm, done_mm,
                                  info_mm.get("event"), alpha_mm)
                    reward_mm += r_mm
                    if ns_mm["in_lava"]:
                        lava_mm += 1
                        if clank_sound: clank_sound.play()
                    if info_mm.get("event") == "mineral":
                        ep_log_mm.minerals_collected += 1
                    ep_log_mm.total_steps = env_mm.step_count
                    ep_log_mm.total_reward = reward_mm
                    ep_log_mm.lava_steps = lava_mm
                    ep_log_mm.minerals_spawned = env_mm.minerals_spawned
                    ep_log_mm.energy_log.append(ns_mm["energy"])
                    ep_log_mm.survived = ns_mm["energy"] > 0
                    ep_log_mm.unsafe_flags.append(in_environment_unsafe_zone(ns_mm))
                    ep_log_mm.srv_flags.append(not mot_in_safe_region(metamo.mot))
                    ep_log_mm.arousal_log.append(mot_arousal(metamo.mot))
                    ep_log_mm.safety_log.append(mot_safety_threshold(metamo.mot))
                    ep_log_mm.individuation_log.append(metamo.mot.G[G_IND])
                    ep_log_mm.transcendence_log.append(metamo.mot.G[G_TRANS])
                    s_mm = ns_mm

            # Auto-advance episode when both done
            if done_bl and done_mm:
                ep_log_bl.total_reward   = reward_bl
                ep_log_bl.lava_steps     = lava_bl
                ep_log_bl.total_steps    = env_bl.step_count
                ep_log_bl.minerals_spawned = env_bl.minerals_spawned
                ep_log_mm.total_reward   = reward_mm
                ep_log_mm.lava_steps     = lava_mm
                ep_log_mm.total_steps    = env_mm.step_count
                ep_log_mm.minerals_spawned = env_mm.minerals_spawned
                bl_metrics.add(ep_log_bl)
                mm_metrics.add(ep_log_mm)

                completed_episodes += 1
                if completed_episodes >= EVAL_EPISODES:
                    evaluation_complete = True
                    paused = True
                else:
                    episode = completed_episodes + 1
                    env_bl, env_mm, s_bl, s_mm = new_episode(episode)
                    done_bl, done_mm, reward_bl, reward_mm, lava_bl, lava_mm, ep_log_bl, ep_log_mm = clear_episode_logs()
                step_accum_ms = 0.0

        # Draw 
        screen.fill(BG)

        # Titles
        t1 = font_lg.render("MetaMo  vs  Baseline RL  —  GridWorld", True, TEXT_COLOR)
        screen.blit(t1, (WIN_W//2 - t1.get_width()//2, 8))

        BL_GRID_X = MARGIN
        MM_GRID_X = BL_GRID_X + GRID_PX + GAP
        DASH_X = MM_GRID_X + GRID_PX + DASH_GAP
        BL_PANEL_Y = GRID_OY
        MM_PANEL_Y = GRID_OY + DASH_PANEL_H + DASH_GAP

        # Baseline grid
        draw_grid(screen, BL_GRID_X, GRID_OY, s_bl,
                  potato_img, mineral_surf, is_metamo=False)

        # MetaMo grid
        draw_grid(screen, MM_GRID_X, GRID_OY, s_mm,
                  potato_img, mineral_surf, is_metamo=True,
                  mot_state=metamo.mot, alpha_dict=alpha_mm)

        # Labels under grids
        bl_lbl = font_title.render("BASELINE RL", True, ACCENT_BL)
        mm_lbl = font_title.render("MetaMo RL",   True, ACCENT_ME)
        screen.blit(bl_lbl, (BL_GRID_X + GRID_PX//2 - bl_lbl.get_width()//2, GRID_OY - 25))
        screen.blit(mm_lbl, (MM_GRID_X + GRID_PX//2 - mm_lbl.get_width()//2, GRID_OY - 25))

        # Panels
        bl_summary = bl_metrics.summary()
        mm_summary = mm_metrics.summary()

        draw_panel(screen, DASH_X, BL_PANEL_Y, DASH_PANEL_H, "Baseline RL", ACCENT_BL,
                   s_bl, episode, completed_episodes, reward_bl,
                   ep_log_bl, bl_summary,
                   font_title, font_xs)

        draw_panel(screen, DASH_X, MM_PANEL_Y, DASH_PANEL_H, "MetaMo RL", ACCENT_ME,
                   s_mm, episode, completed_episodes, reward_mm,
                   ep_log_mm, mm_summary,
                   font_title, font_xs,
                   mot_state=metamo.mot, alpha_dict=alpha_mm)

        # Footer / controls
        ctrl = font_xs.render(
            "SPACE: pause   R: reset   F/UP: faster   S/DOWN: slower   Q: quit"
            f"   |   Speed: {steps_per_second:.0f} steps/s   |   Episodes completed: {completed_episodes}/{EVAL_EPISODES}",
            True, DIM_TEXT)
        screen.blit(ctrl, (WIN_W//2 - ctrl.get_width()//2, WIN_H - 22))

        if evaluation_complete:
            pt = font_lg.render("EVALUATION COMPLETE", True, (255, 220, 60))
            screen.blit(pt, (WIN_W//2 - pt.get_width()//2, WIN_H//2 - 20))
        elif paused:
            pt = font_lg.render("PAUSED", True, (255, 220, 60))
            screen.blit(pt, (WIN_W//2 - pt.get_width()//2, WIN_H//2 - 20))

        pygame.display.flip()

    print("\n" + "="*55)
    for mc in [bl_metrics, mm_metrics]:
        s = mc.summary()
        if not s:
            continue
        print(f"\n[ {s['label']} ]  ({s['n_episodes']} episodes)")
        print(f"  Completion rate : {s['completion_rate']['mean']:.3f} ± {s['completion_rate']['std']:.3f}")
        print(f"  Lava rate       : {s['lava_rate']['mean']:.3f} ± {s['lava_rate']['std']:.3f}")
        print(f"  SRV rate        : {s['srv_rate']['mean']:.3f} ± {s['srv_rate']['std']:.3f}")
        print(f"  Unsafe-zone rate: {s['unsafe_rate']['mean']:.3f} ± {s['unsafe_rate']['std']:.3f}")
        print(f"  Recovery time   : {s['recovery_time']['mean']:.1f} ± {s['recovery_time']['std']:.1f}")
        print(f"  Total reward    : {s['total_reward']['mean']:.1f} ± {s['total_reward']['std']:.1f}")
    print("="*55)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

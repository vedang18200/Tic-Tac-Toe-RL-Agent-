import streamlit as st
import numpy as np
import random
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Tic-Tac-Toe RL Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0d0d0f;
    --surface: #16161a;
    --border: #2a2a32;
    --accent: #00ff88;
    --accent2: #ff4d6d;
    --text: #e8e8f0;
    --muted: #5a5a72;
    --x-color: #ff4d6d;
    --o-color: #00ff88;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

.stApp { background-color: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Sliders */
.stSlider > div > div > div > div { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
    transform: translateY(-1px) !important;
}

/* Cell buttons */
.cell-btn > button {
    width: 90px !important;
    height: 90px !important;
    font-size: 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
    border: 2px solid var(--border) !important;
    background: var(--surface) !important;
    transition: all 0.15s ease !important;
}
.cell-btn > button:hover {
    border-color: var(--muted) !important;
    background: #1e1e26 !important;
    transform: scale(1.04) !important;
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-playing { background: #1a2e1a; color: var(--accent); border: 1px solid var(--accent); }
.badge-win     { background: #1a2e1a; color: var(--accent); border: 1px solid var(--accent); }
.badge-lose    { background: #2e1a1e; color: var(--accent2); border: 1px solid var(--accent2); }
.badge-draw    { background: #1e1e26; color: #aaa; border: 1px solid #555; }

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* Q-value heatmap cells */
.q-cell {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 70px; height: 70px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 2px;
    border: 1px solid var(--border);
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Q-LEARNING AGENT
# ─────────────────────────────────────────
class TicTacToeEnv:
    def reset(self):
        return tuple([0] * 9)

    def available_actions(self, state):
        return [i for i, v in enumerate(state) if v == 0]

    def step(self, state, action, player):
        s = list(state)
        s[action] = player
        state = tuple(s)
        winner = self.check_winner(state)
        done = winner != 0 or len(self.available_actions(state)) == 0
        reward = 0
        if winner == player:
            reward = 1
        elif winner == -player:
            reward = -1
        return state, reward, done, winner

    def check_winner(self, state):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a,b,c in wins:
            if state[a] == state[b] == state[c] != 0:
                return state[a]
        return 0


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(available)
        q_vals = [self.get_q(state, a) for a in available]
        max_q = max(q_vals)
        best = [a for a, q in zip(available, q_vals) if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_available, done):
        current_q = self.get_q(state, action)
        if done or not next_available:
            target = reward
        else:
            target = reward + self.gamma * max([self.get_q(next_state, a) for a in next_available])
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

    def get_q_values_for_state(self, state, available):
        return {a: self.get_q(state, a) for a in range(9)}


def train_agent(episodes, alpha, gamma, epsilon, progress_bar, status_text):
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    wins, losses, draws = [], [], []
    win_count = loss_count = draw_count = 0
    log_every = max(1, episodes // 100)

    for ep in range(episodes):
        state = env.reset()
        player = 1
        done = False

        transitions = []

        while not done:
            available = env.available_actions(state)
            if not available:
                break
            action = agent.choose_action(state, available, greedy=False)
            next_state, reward, done, winner = env.step(state, action, player)
            next_avail = env.available_actions(next_state)
            transitions.append((state, action, reward, next_state, next_avail, done, player))
            state = next_state
            player = -player

        # Assign rewards and update
        for (s, a, r, ns, na, d, p) in transitions:
            final_winner = winner
            if final_winner == p:
                final_r = 1
            elif final_winner == -p:
                final_r = -1
            else:
                final_r = 0.1 if d else 0
            agent.update(s, a, final_r, ns, na, d)

        if winner == 1:
            win_count += 1
        elif winner == -1:
            loss_count += 1
        else:
            draw_count += 1

        if (ep + 1) % log_every == 0:
            wins.append(win_count / (ep + 1) * 100)
            losses.append(loss_count / (ep + 1) * 100)
            draws.append(draw_count / (ep + 1) * 100)
            progress_bar.progress((ep + 1) / episodes)
            status_text.text(f"Episode {ep+1}/{episodes} — W:{win_count} L:{loss_count} D:{draw_count}")

    return agent, wins, losses, draws


# ─────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────
defaults = {
    "agent": None,
    "board": [0] * 9,
    "game_over": False,
    "winner": 0,
    "human_symbol": 1,
    "current_player": 1,
    "score": {"wins": 0, "losses": 0, "draws": 0},
    "training_done": False,
    "train_wins": [],
    "train_losses": [],
    "train_draws": [],
    "status_msg": "Train the agent first, then play!",
    "status_type": "playing",
    "total_games": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
env = TicTacToeEnv()

def check_winner_local(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0

def ai_move():
    if st.session_state.agent is None:
        return
    state = tuple(st.session_state.board)
    available = [i for i, v in enumerate(state) if v == 0]
    if not available:
        return
    action = st.session_state.agent.choose_action(state, available, greedy=True)
    st.session_state.board[action] = -st.session_state.human_symbol
    winner = check_winner_local(st.session_state.board)
    if winner != 0:
        st.session_state.game_over = True
        st.session_state.winner = winner
        if winner == st.session_state.human_symbol:
            st.session_state.score["wins"] += 1
            st.session_state.status_msg = "🎉 You won!"
            st.session_state.status_type = "win"
        else:
            st.session_state.score["losses"] += 1
            st.session_state.status_msg = "🤖 AI wins!"
            st.session_state.status_type = "lose"
        st.session_state.total_games += 1
    elif all(v != 0 for v in st.session_state.board):
        st.session_state.game_over = True
        st.session_state.winner = 0
        st.session_state.score["draws"] += 1
        st.session_state.status_msg = "🤝 It's a draw!"
        st.session_state.status_type = "draw"
        st.session_state.total_games += 1

def handle_click(idx):
    if st.session_state.game_over:
        return
    if st.session_state.agent is None:
        st.session_state.status_msg = "⚠️ Train the agent first!"
        return
    if st.session_state.board[idx] != 0:
        return
    st.session_state.board[idx] = st.session_state.human_symbol
    winner = check_winner_local(st.session_state.board)
    if winner != 0:
        st.session_state.game_over = True
        st.session_state.winner = winner
        st.session_state.score["wins"] += 1
        st.session_state.status_msg = "🎉 You won! Nice move!"
        st.session_state.status_type = "win"
        st.session_state.total_games += 1
        return
    if all(v != 0 for v in st.session_state.board):
        st.session_state.game_over = True
        st.session_state.score["draws"] += 1
        st.session_state.status_msg = "🤝 It's a draw!"
        st.session_state.status_type = "draw"
        st.session_state.total_games += 1
        return
    ai_move()

def reset_game():
    st.session_state.board = [0] * 9
    st.session_state.game_over = False
    st.session_state.winner = 0
    st.session_state.current_player = 1
    st.session_state.status_msg = "Your turn! Click a cell."
    st.session_state.status_type = "playing"


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Agent Config")
    st.markdown('<div class="section-title">Hyperparameters</div>', unsafe_allow_html=True)

    episodes = st.slider("Training Episodes", 1000, 100000, 20000, step=1000,
                         help="More episodes = smarter agent")
    alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.5, step=0.01,
                      help="How fast the agent updates its Q-values")
    gamma = st.slider("Discount Factor (γ)", 0.5, 1.0, 0.95, step=0.01,
                      help="How much future rewards matter")
    epsilon = st.slider("Exploration (ε)", 0.01, 1.0, 0.2, step=0.01,
                        help="Probability of random move during training")

    st.markdown("---")
    train_btn = st.button("🚀 Train Agent", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Play Settings</div>', unsafe_allow_html=True)
    symbol = st.radio("Play as", ["X (first)", "O (second)"])
    st.session_state.human_symbol = 1 if "X" in symbol else -1

    st.markdown("---")
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.caption("Q-Learning agent trained via self-play. State space: 3⁹ = 19,683 possible boards. Action space: up to 9 moves per state.")


# ─────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────
if train_btn:
    st.session_state.training_done = False
    st.session_state.agent = None
    with st.spinner(""):
        prog = st.progress(0)
        status = st.empty()
        agent, w, l, d = train_agent(episodes, alpha, gamma, epsilon, prog, status)
        st.session_state.agent = agent
        st.session_state.train_wins = w
        st.session_state.train_losses = l
        st.session_state.train_draws = d
        st.session_state.training_done = True
        st.session_state.score = {"wins": 0, "losses": 0, "draws": 0}
        st.session_state.total_games = 0
        prog.empty()
        status.empty()
        reset_game()
        st.session_state.status_msg = f"Agent trained on {episodes:,} episodes. Your turn!"
    st.rerun()


# ─────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────
st.markdown("# Tic-Tac-Toe &nbsp;·&nbsp; RL Agent")
st.markdown('<p style="color:var(--muted);font-family:Space Mono,monospace;font-size:0.8rem;margin-top:-0.5rem;">Q-Learning · Self-Play Training · Human vs AI</p>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

# ── LEFT: GAME BOARD ──
with left:
    st.markdown('<div class="section-title">Game Board</div>', unsafe_allow_html=True)

    # Status badge
    badge_cls = f"badge-{st.session_state.status_type}"
    st.markdown(f'<div style="margin-bottom:1rem"><span class="status-badge {badge_cls}">{st.session_state.status_msg}</span></div>',
                unsafe_allow_html=True)

    # Board grid
    symbol_map = {1: "✕", -1: "○", 0: ""}
    color_map  = {1: "#ff4d6d", -1: "#00ff88", 0: "#e8e8f0"}

    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            idx = row * 3 + col
            val = st.session_state.board[idx]
            sym = symbol_map[val]
            clr = color_map[val]

            with cols[col]:
                label = f":{sym}" if sym else " "

                # Style by symbol
                if val == 1:
                    btn_label = "✕"
                elif val == -1:
                    btn_label = "○"
                else:
                    btn_label = " "

                disabled = (val != 0) or st.session_state.game_over or (st.session_state.agent is None)

                if st.button(btn_label, key=f"cell_{idx}", disabled=disabled, use_container_width=True):
                    handle_click(idx)
                    st.rerun()

    # Action buttons
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 New Game", use_container_width=True):
            reset_game()
            # If AI goes first
            if st.session_state.human_symbol == -1 and st.session_state.agent:
                ai_move()
            st.rerun()
    with c2:
        if st.button("🗑️ Reset Score", use_container_width=True):
            st.session_state.score = {"wins": 0, "losses": 0, "draws": 0}
            st.session_state.total_games = 0
            reset_game()
            st.rerun()

    st.markdown("")

    # Score cards
    s = st.session_state.score
    total = max(1, st.session_state.total_games)
    sc1, sc2, sc3 = st.columns(3)
    metrics = [
        (sc1, s["wins"],   "var(--o-color)", "Wins"),
        (sc2, s["losses"], "var(--x-color)", "Losses"),
        (sc3, s["draws"],  "#aaa",           "Draws"),
    ]
    for col, val, clr, lbl in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{clr}">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Win rate bar
    if st.session_state.total_games > 0:
        wr = s["wins"] / total * 100
        st.markdown(f"""
        <div style="margin-top:1rem">
            <div style="display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:0.7rem;color:var(--muted);margin-bottom:4px">
                <span>WIN RATE</span><span style="color:var(--accent)">{wr:.0f}%</span>
            </div>
            <div style="background:var(--border);border-radius:4px;height:6px">
                <div style="background:var(--accent);width:{wr}%;height:6px;border-radius:4px;transition:width 0.4s"></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ── RIGHT: TRAINING STATS + Q-VALUES ──
with right:

    # Training chart
    st.markdown('<div class="section-title">Training Progress</div>', unsafe_allow_html=True)

    if st.session_state.training_done and st.session_state.train_wins:
        w = st.session_state.train_wins
        l = st.session_state.train_losses
        d = st.session_state.train_draws
        x = list(range(1, len(w) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=w, name="Win %", line=dict(color="#00ff88", width=2),
                                  fill="tozeroy", fillcolor="rgba(0,255,136,0.07)"))
        fig.add_trace(go.Scatter(x=x, y=l, name="Loss %", line=dict(color="#ff4d6d", width=2),
                                  fill="tozeroy", fillcolor="rgba(255,77,109,0.07)"))
        fig.add_trace(go.Scatter(x=x, y=d, name="Draw %", line=dict(color="#aaa", width=1.5, dash="dot")))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono, monospace", color="#5a5a72", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f0")),
            xaxis=dict(showgrid=False, color="#2a2a32", title="Episode (×log)", title_font=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1e1e26", color="#2a2a32", title="%", title_font=dict(size=10), range=[0, 100]),
            height=240,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Final stats
        fa, fb, fc = st.columns(3)
        final_metrics = [
            (fa, f"{w[-1]:.1f}%", "Final Win %",  "var(--o-color)"),
            (fb, f"{l[-1]:.1f}%", "Final Loss %", "var(--x-color)"),
            (fc, f"{d[-1]:.1f}%", "Final Draw %", "#aaa"),
        ]
        for col, val, lbl, clr in final_metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="padding:0.7rem">
                    <div class="metric-val" style="color:{clr};font-size:1.4rem">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # Q-table size
        q_size = len(st.session_state.agent.q_table)
        st.markdown(f'<p style="font-family:Space Mono,monospace;font-size:0.7rem;color:var(--muted);margin-top:0.8rem">Q-TABLE ENTRIES: <span style="color:var(--accent)">{q_size:,}</span> state-action pairs learned</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--surface);border:1px dashed var(--border);border-radius:10px;
                    padding:2rem;text-align:center;height:240px;display:flex;align-items:center;justify-content:center;flex-direction:column">
            <div style="font-size:2rem;margin-bottom:0.5rem">📊</div>
            <div style="font-family:Space Mono,monospace;font-size:0.75rem;color:var(--muted)">
                Train the agent to see<br>performance curves here
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Q-Value heatmap for current board
    st.markdown('<div class="section-title">AI Q-Values (Current Board)</div>', unsafe_allow_html=True)

    if st.session_state.agent:
        state = tuple(st.session_state.board)
        available = [i for i, v in enumerate(state) if v == 0]
        q_vals = st.session_state.agent.get_q_values_for_state(state, available)

        if q_vals and available:
            vals = [q_vals.get(i, None) for i in range(9)]
            avail_vals = [q_vals[i] for i in available]
            mn = min(avail_vals) if avail_vals else 0
            mx = max(avail_vals) if avail_vals else 1
            rng = mx - mn if mx != mn else 1

            def q_to_color(q, is_avail):
                if not is_avail:
                    return "#1a1a20", "#444"
                t = (q - mn) / rng
                r = int(255 * t + 77 * (1 - t))
                g = int(255 * (1 - t) + 77 * t)
                b = int(100 * (1 - t) + 109 * t)
                return f"rgba({r},{g},{b},0.25)", f"rgba({r},{g},{b},0.8)"

            html = '<div style="display:grid;grid-template-columns:repeat(3,76px);gap:4px;margin-top:0.5rem">'
            best_action = max(available, key=lambda a: q_vals[a]) if available else -1

            for i in range(9):
                is_avail = i in available
                q = q_vals.get(i, 0)
                bg, border = q_to_color(q, is_avail)
                marker = "★" if i == best_action else ""
                val_txt = f"{q:.3f}" if is_avail else "—"
                board_sym = {1: "✕", -1: "○", 0: ""}[st.session_state.board[i]]

                if board_sym:
                    color = "#ff4d6d" if st.session_state.board[i] == 1 else "#00ff88"
                    html += f'<div style="width:76px;height:76px;background:var(--surface);border:2px solid {border};border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;font-weight:800;color:{color}">{board_sym}</div>'
                else:
                    html += f'<div style="width:76px;height:76px;background:{bg};border:1.5px solid {border};border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px"><span style="font-family:Space Mono,monospace;font-size:0.68rem;font-weight:700;color:#e8e8f0">{val_txt}</span><span style="font-size:0.7rem;color:gold">{marker}</span></div>'

            html += '</div>'
            html += '<p style="font-family:Space Mono,monospace;font-size:0.65rem;color:var(--muted);margin-top:0.6rem">★ = AI\'s best move &nbsp;·&nbsp; Green = high value &nbsp;·&nbsp; Red = low value</p>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--muted);font-family:Space Mono,monospace;font-size:0.8rem">Game over — start a new game to see Q-values.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-family:Space Mono,monospace;font-size:0.8rem">Train the agent to see Q-values for the current board state.</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────
#  HOW TO PLAY
# ─────────────────────────────────────────
st.markdown("---")
with st.expander("📖 How to Use This App"):
    st.markdown("""
**1. Configure** hyperparameters in the sidebar (defaults work great to start)

**2. Train** — Click **🚀 Train Agent**. The agent plays thousands of games against itself using Q-Learning.

**3. Play** — Click any empty cell on the board to make your move. The AI responds instantly.

**4. Analyze** — Watch the **Q-Value heatmap** update in real time to see how the AI evaluates each position. The ★ marks its preferred move.

---
**Q-Learning Basics:**
- The agent maintains a table of `Q(state, action)` values
- Higher Q = AI prefers that cell in the current board state
- During training: ε-greedy exploration (random moves with probability ε)
- During play: purely greedy (always picks highest Q-value)
    """)

# main_3d_online_png.py
"""
3D Chess (PNG billboards) + Online multiplayer + Smooth animations.

Controls:
 - Left-click: select/move
 - Right-drag: rotate camera
 - Mouse wheel: zoom
 - U: undo last player+AI moves (local mode only)
 - R: restart board
 - O: connect to server (matchmaking)
 - P: disconnect / leave online mode
"""

import os
import time
import queue
import json
import threading
import asyncio
import pygame
import websockets
import numpy as np
import chess
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from stockfish import Stockfish

# ---------------- Config ----------------
WIDTH, HEIGHT = 1000, 700
SQUARE = 1.0
AI_DELAY = 0.5
ANIM_DURATION = 0.40
STOCKFISH_PATH = r"C:\Users\Hariharan\Desktop\python\chess_project\stockfish\stockfish-windows-x86-64-avx2.exe"
IMAGE_FOLDER = r"C:\Users\Hariharan\Desktop\python\chess_project\images"
SERVER_URI = "ws://127.0.0.1:8765"  # change to server address if needed

HIGHLIGHT_COLOR = (0.0, 1.0, 0.0, 0.35)
MOVE_COLOR = (1.0, 1.0, 0.0, 0.30)
CHECK_COLOR = (1.0, 0.0, 0.0, 0.5)

# ---------------- Pygame + OpenGL init ----------------
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Chess (PNG pieces, Online)")
clock = pygame.time.Clock()

glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glClearColor(0.2, 0.22, 0.25, 1.0)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, WIDTH / HEIGHT, 0.1, 100.0)
glMatrixMode(GL_MODELVIEW)

# Camera spherical coords
cam_distance = 12.0
cam_rot_x, cam_rot_y = 25.0, -30.0

# ---------------- Chess & AI ----------------
board = chess.Board()
stockfish = Stockfish(path=STOCKFISH_PATH)
last_ai_time = 0.0
move_history = []

# ---------------- WebSocket client ----------------
ws_send_q = queue.Queue()
ws_recv_q = queue.Queue()
online_mode = False
my_color = None
stop_ws_flag = threading.Event()

def start_ws_thread():
    def runner():
        asyncio.run(ws_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t

async def ws_main():
    global online_mode, my_color
    stop_ws_flag.clear()
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            await websocket.send(json.dumps({"type": "join"}))
            online_mode = True
            print("[ws] connected, waiting for match...")
            async def sender():
                while not stop_ws_flag.is_set():
                    try:
                        msg = ws_send_q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    await websocket.send(json.dumps(msg))
                try:
                    await websocket.close()
                except:
                    pass
            async def receiver():
                while not stop_ws_flag.is_set():
                    try:
                        raw = await websocket.recv()
                    except websockets.ConnectionClosed:
                        break
                    data = json.loads(raw)
                    ws_recv_q.put(data)
            await asyncio.gather(sender(), receiver())
    except Exception as e:
        print("[ws] exception:", e)
    finally:
        online_mode = False
        my_color = None
        print("[ws] disconnected")

def stop_ws_thread():
    stop_ws_flag.set()

# ---------------- Load PNG textures for pieces ----------------
piece_textures = {}  # key e.g. 'wP' -> tex_id, (w,h)
def load_piece_textures():
    pieces = ['P','R','N','B','Q','K']
    for color in ['w','b']:
        for p in pieces:
            fname = os.path.join(IMAGE_FOLDER, f"{color}{p}.png")
            if not os.path.exists(fname):
                print("Missing image:", fname)
                continue
            surf = pygame.image.load(fname).convert_alpha()
            w, h = surf.get_size()
            tex_data = pygame.image.tostring(surf, "RGBA", 1)
            tid = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
            glBindTexture(GL_TEXTURE_2D, 0)
            piece_textures[color + p] = (tid, w, h)

load_piece_textures()

# ---------------- Animation state ----------------
anim_state = None  # dict or None

def square_center(col, row):
    x = (col - 3.5) * SQUARE
    z = (row - 3.5) * SQUARE
    return x, z

def ease_in_out(t):
    return t*t*(3 - 2*t)

def start_move_animation(move, duration=ANIM_DURATION):
    global anim_state
    # move is chess.Move and should be pushed to board BEFORE calling this
    from_col, from_row = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    to_col, to_row = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    fx, fz = square_center(from_col, from_row)
    tx, tz = square_center(to_col, to_row)
    piece = board.piece_at(move.to_square)
    if piece is None:
        # fallback (shouldn't happen since we call after push)
        piece_symbol = move.promotion and 'Q' or 'P'
        piece_white = True
    else:
        piece_symbol = piece.symbol().upper()
        piece_white = (piece.color == chess.WHITE)
    anim_state = {
        "move": move,
        "symbol": piece_symbol,
        "white": piece_white,
        "from": (fx, fz),
        "to": (tx, tz),
        "t": 0.0,
        "duration": duration,
        "hide_from_sq": move.from_square,
        "hide_to_sq": move.to_square,
    }

def update_animation(dt):
    global anim_state
    if not anim_state:
        return
    anim_state["t"] += dt
    if anim_state["t"] >= anim_state["duration"]:
        anim_state = None

# ---------------- Drawing helpers ----------------
def draw_board():
    for r in range(8):
        for c in range(8):
            x = (c - 3.5) * SQUARE
            z = (r - 3.5) * SQUARE
            if (r + c) % 2 == 0:
                glColor3f(0.94, 0.85, 0.70)
            else:
                glColor3f(0.71, 0.53, 0.39)
            glBegin(GL_QUADS)
            glVertex3f(x - SQUARE/2, 0.0, z - SQUARE/2)
            glVertex3f(x + SQUARE/2, 0.0, z - SQUARE/2)
            glVertex3f(x + SQUARE/2, 0.0, z + SQUARE/2)
            glVertex3f(x - SQUARE/2, 0.0, z + SQUARE/2)
            glEnd()

def draw_highlight(x, z, color, height=0.01):
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex3f(x - SQUARE/2, height, z - SQUARE/2)
    glVertex3f(x + SQUARE/2, height, z - SQUARE/2)
    glVertex3f(x + SQUARE/2, height, z + SQUARE/2)
    glVertex3f(x - SQUARE/2, height, z + SQUARE/2)
    glEnd()

def draw_piece_png(key, x, z, size=0.45):
    # key e.g. 'wP'. draws textured quad facing camera (simple upright quad)
    if key not in piece_textures:
        return
    tid, w, h = piece_textures[key]
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tid)
    # compute quad half-size relative to SQUARE
    hs = size * SQUARE / 2.0
    y = 0.02  # slight lift above board
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(x - hs, y, z - hs)
    glTexCoord2f(1, 0); glVertex3f(x + hs, y, z - hs)
    glTexCoord2f(1, 1); glVertex3f(x + hs, y, z + hs)
    glTexCoord2f(0, 1); glVertex3f(x - hs, y, z + hs)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

# ---------------- Picking helpers ----------------
def screen_ray(mx, my):
    viewport = glGetIntegerv(GL_VIEWPORT)
    proj = glGetDoublev(GL_PROJECTION_MATRIX)
    model = glGetDoublev(GL_MODELVIEW_MATRIX)
    nx, ny = mx, viewport[3] - my
    near = gluUnProject(nx, ny, 0.0, model, proj, viewport)
    far = gluUnProject(nx, ny, 1.0, model, proj, viewport)
    near = np.array(near); far = np.array(far)
    dir = far - near
    dir /= np.linalg.norm(dir)
    return near, dir

def ray_plane(orig, dir, y=0.0):
    if abs(dir[1]) < 1e-6:
        return None
    t = (y - orig[1]) / dir[1]
    if t < 0:
        return None
    return orig + dir * t

def pick_square(mx, my):
    near, dir = screen_ray(mx, my)
    inter = ray_plane(near, dir, 0.0)
    if inter is None:
        return None
    xz = inter[[0,2]]
    col = int(round((xz[0] / SQUARE) + 3.5))
    row = int(round((xz[1] / SQUARE) + 3.5))
    if 0 <= col < 8 and 0 <= row < 8:
        return col, row
    return None

# ---------------- AI local ----------------
def ai_move_local():
    global last_ai_time
    # AI only when not online and not animating
    if online_mode or anim_state:
        return
    if board.turn == chess.BLACK and time.time() - last_ai_time > AI_DELAY:
        stockfish.set_fen_position(board.fen())
        best = stockfish.get_best_move()
        if best:
            mv = chess.Move.from_uci(best)
            if mv in board.legal_moves:
                board.push(mv)
                move_history.append(mv)
                start_move_animation(mv)
        last_ai_time = time.time()

# ---------------- Main loop variables ----------------
selected_square = None
rotating = False

# ---------------- Main loop ----------------
running = True
while running:
    dt = clock.tick(60) / 1000.0

    # --- Process incoming websocket messages ---
    try:
        while True:
            msg = ws_recv_q.get_nowait()
            mtype = msg.get("type")
            if mtype == "status":
                print("[server]", msg.get("message"))
            elif mtype == "start":
                my_color = msg.get("color")
                print("[server] matched. You are:", my_color)
            elif mtype == "move":
                uci = msg.get("uci")
                if uci:
                    mv = chess.Move.from_uci(uci)
                    if mv in board.legal_moves:
                        board.push(mv)
                        move_history.append(mv)
                        start_move_animation(mv)
                    else:
                        print("[ws] received illegal move:", uci)
            elif mtype == "opponent_disconnected":
                print("[ws] opponent disconnected; leaving online mode")
                online_mode = False
                my_color = None
            else:
                print("[ws] msg:", msg)
    except queue.Empty:
        pass

    # --- Events ---
    for ev in pygame.event.get():
        if ev.type == QUIT:
            running = False
        elif ev.type == KEYDOWN:
            if ev.key == pygame.K_u and move_history and not online_mode and not anim_state:
                # undo last player+AI moves
                try:
                    board.pop(); move_history.pop()
                    board.pop(); move_history.pop()
                except:
                    pass
            elif ev.key == pygame.K_r and not anim_state:
                board.reset(); move_history.clear()
            elif ev.key == pygame.K_o and not online_mode:
                print("[ws] connecting to", SERVER_URI)
                ws_thread = start_ws_thread()
            elif ev.key == pygame.K_p and online_mode:
                print("[ws] disconnecting")
                stop_ws_thread()
        elif ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1:
                # if animating, ignore piece input
                if anim_state:
                    continue
                pick = pick_square(ev.pos[0], ev.pos[1])
                if not pick:
                    continue
                col, row = pick
                sq = chess.square(col, 7 - row)
                # enforce allowed move in online mode
                if online_mode and my_color:
                    allowed = (board.turn == chess.WHITE and my_color == "white") or (board.turn == chess.BLACK and my_color == "black")
                    if not allowed:
                        # not our turn
                        continue
                # selection / move
                if selected_square is None:
                    p = board.piece_at(sq)
                    if p and p.color == board.turn:
                        selected_square = sq
                else:
                    mv = chess.Move(selected_square, sq)
                    # handle promotion auto-queen for simplicity
                    if board.piece_type_at(selected_square) == chess.PAWN and chess.square_rank(sq) in [0, 7]:
                        mv.promotion = chess.QUEEN
                    if mv in board.legal_moves:
                        # push locally and animate; if online send to server as well
                        board.push(mv)
                        move_history.append(mv)
                        start_move_animation(mv)
                        if online_mode and my_color:
                            ws_send_q.put({"type": "move", "uci": mv.uci()})
                            # no AI in online mode
                        else:
                            # local vs AI: delay AI until after animation
                            last_ai_time = time.time() + ANIM_DURATION
                    selected_square = None
            elif ev.button == 3:
                rotating = True
                pygame.mouse.get_rel()
            elif ev.button == 4:
                cam_distance = max(6.0, cam_distance - 0.6)
            elif ev.button == 5:
                cam_distance = min(24.0, cam_distance + 0.6)
        elif ev.type == MOUSEBUTTONUP:
            if ev.button == 3:
                rotating = False
        elif ev.type == MOUSEMOTION:
            if rotating and ev.rel:
                dx, dy = ev.rel
                cam_rot_y += dx * 0.3
                cam_rot_x += dy * 0.3

    # --- Update animation ---
    update_animation(dt)

    # --- Camera / Clear ---
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    rx = np.radians(cam_rot_x)
    ry = np.radians(cam_rot_y)
    cam_x = cam_distance * np.cos(rx) * np.sin(ry)
    cam_y = cam_distance * np.sin(rx)
    cam_z = cam_distance * np.cos(rx) * np.cos(ry)
    gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0)

    # --- Draw board ---
    draw_board()

    # --- Highlights ---
    if selected_square is not None:
        f = chess.square_file(selected_square)
        r = chess.square_rank(selected_square)
        x, z = square_center(f, r)
        draw_highlight(x, z, HIGHLIGHT_COLOR)
        for mv in board.legal_moves:
            if mv.from_square == selected_square:
                f2 = chess.square_file(mv.to_square)
                r2 = chess.square_rank(mv.to_square)
                x2, z2 = square_center(f2, r2)
                draw_highlight(x2, z2, MOVE_COLOR)

    if board.is_check():
        ks = board.king(board.turn)
        if ks is not None:
            f, r = chess.square_file(ks), chess.square_rank(ks)
            xk, zk = square_center(f, r)
            draw_highlight(xk, zk, CHECK_COLOR)

    # --- Draw pieces (skip hidden squares while animating) ---
    hide_from = anim_state.get("hide_from_sq") if anim_state else None
    hide_to = anim_state.get("hide_to_sq") if anim_state else None
    for r in range(8):
        for c in range(8):
            sq = chess.square(c, r)
            if anim_state and (sq == hide_from or sq == hide_to):
                continue
            p = board.piece_at(sq)
            if p:
                key = ('w' if p.color == chess.WHITE else 'b') + p.symbol().upper()
                x, z = square_center(c, r)
                draw_piece_png(key, x, z)

    # --- Draw animated piece on top (if any) ---
    if anim_state:
        t = anim_state["t"] / anim_state["duration"]
        t = max(0.0, min(1.0, t))
        te = ease_in_out(t)
        fx, fz = anim_state["from"]
        tx, tz = anim_state["to"]
        cx = fx + (tx - fx) * te
        cz = fz + (tz - fz) * te
        # draw the animating piece texture
        key = ('w' if anim_state["white"] else 'b') + anim_state["symbol"]
        draw_piece_png(key, cx, cz)

    # --- AI local move (only when not online) ---
    if not online_mode:
        ai_move_local()

    pygame.display.flip()

# cleanup
if online_mode:
    stop_ws_thread()
pygame.quit()

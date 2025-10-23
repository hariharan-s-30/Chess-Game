# ==================== CHESS AI BACKEND API ====================
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import chess.engine
import numpy as np
import tensorflow as tf
import json
import os
from typing import Optional, Dict, List
import uuid
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Chess AI API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key for security
API_KEY = "mysecretkey123"

# Request models
class MoveRequest(BaseModel):
    move: str
    difficulty: int

class DifficultyRequest(BaseModel):
    difficulty: int

class ResetRequest(BaseModel):
    difficulty: int

class UndoRequest(BaseModel):
    difficulty: int

# Response models
class MoveResponse(BaseModel):
    next_move: Optional[str] = None
    game_over: bool = False
    result: Optional[str] = None
    fen: str
    message: str

class BoardResponse(BaseModel):
    board_fen: str
    current_turn: str
    game_over: bool = False
    result: Optional[str] = None

class DifficultyResponse(BaseModel):
    message: str
    current_difficulty: int
    elo_estimate: int

# Chess AI Engine with Difficulty
class ChessAIEngine:
    def __init__(self, model_path: str, move_dict_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(move_dict_path, 'r') as f:
            self.MOVE2IDX = json.load(f)
        self.IDX2MOVE = {v: k for k, v in self.MOVE2IDX.items()}
        
        # Game state storage
        self.game_states: Dict[str, chess.Board] = {}
        
        # Difficulty mapping (ELO to AI parameters)
        self.difficulty_settings = {
            250: {'temperature': 2.5, 'top_moves': 15, 'blunder_chance': 0.4, 'search_depth': 1},
            500: {'temperature': 2.0, 'top_moves': 12, 'blunder_chance': 0.3, 'search_depth': 1},
            750: {'temperature': 1.5, 'top_moves': 10, 'blunder_chance': 0.2, 'search_depth': 1},
            1000: {'temperature': 1.2, 'top_moves': 8, 'blunder_chance': 0.15, 'search_depth': 2},
            1250: {'temperature': 1.0, 'top_moves': 6, 'blunder_chance': 0.1, 'search_depth': 2},
            1500: {'temperature': 0.7, 'top_moves': 4, 'blunder_chance': 0.05, 'search_depth': 2},
            1750: {'temperature': 0.4, 'top_moves': 3, 'blunder_chance': 0.02, 'search_depth': 3},
            2000: {'temperature': 0.1, 'top_moves': 2, 'blunder_chance': 0.0, 'search_depth': 3}
        }
    
    def create_session(self) -> str:
        """Create a new game session"""
        session_id = str(uuid.uuid4())
        self.game_states[session_id] = chess.Board()
        return session_id
        
    def encode_board(self, board: chess.Board) -> np.ndarray:
        """Encode the board as 12×8×8 planes (flattened to 768) for NN input."""
        # Planes order: [white P,N,B,R,Q,K, black p,n,b,r,q,k]
        piece_planes = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        encoded = np.zeros((12, 8, 8), dtype=np.float32)

        for plane_idx, piece_symbol in enumerate(piece_planes):
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.symbol() == piece_symbol:
                    row = 7 - chess.square_rank(square)   # flip rank (a1→bottom)
                    col = chess.square_file(square)
                    encoded[plane_idx, row, col] = 1.0

        # Flatten to 1D array of length 768
        return encoded.flatten()

        
        for char in board_fen:
            if char == '/':
                continue
            elif char.isdigit():
                square_index += int(char)
            else:
                board_arr[square_index] = piece_to_num.get(char, 0)
                square_index += 1
        
        return board_arr
    
    def get_ai_move(self, board: chess.Board, difficulty: int) -> Optional[str]:
        """Get AI move based on difficulty level"""
        if difficulty not in self.difficulty_settings:
            # Find closest difficulty
            difficulties = list(self.difficulty_settings.keys())
            closest = min(difficulties, key=lambda x: abs(x - difficulty))
            settings = self.difficulty_settings[closest]
        else:
            settings = self.difficulty_settings[difficulty]
        
        # Encode current position
        position_encoded = self.encode_board(board)
        position_batch = np.array([position_encoded])
        
        # Get neural network predictions
        predictions = self.model.predict(position_batch, verbose=0)[0]
        
        # Apply temperature for randomness
        temperature = settings['temperature']
        if temperature != 1.0:
            predictions = self._apply_temperature(predictions, temperature)
        
        # Get legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_move_indices = [self.MOVE2IDX[move] for move in legal_moves if move in self.MOVE2IDX]
        
        if not legal_move_indices:
            return None
        
        # Filter predictions to legal moves only
        legal_predictions = np.zeros(len(self.MOVE2IDX))
        legal_predictions[legal_move_indices] = predictions[legal_move_indices]
        
        # Apply blunder chance for lower difficulties
        if np.random.random() < settings['blunder_chance']:
            blunder_move = self._make_blunder_move(board, legal_moves)
            if blunder_move:
                return blunder_move.uci()
        
        # Get top moves based on difficulty
        top_k = settings['top_moves']
        top_indices = np.argsort(legal_predictions)[-top_k:][::-1]
        
        # Choose from top moves (with randomness for lower difficulties)
        if len(top_indices) > 1 and settings['temperature'] > 0.5:
            weights = legal_predictions[top_indices]
            weights = weights / np.sum(weights)  # Normalize
            chosen_idx = np.random.choice(top_indices, p=weights)
        else:
            chosen_idx = top_indices[0]  # Best move
        
        move_uci = self.IDX2MOVE[chosen_idx]
        return move_uci
    
    def _apply_temperature(self, predictions: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature to softmax outputs for randomness"""
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions - np.max(predictions))
        return exp_preds / np.sum(exp_preds)
    
    def _make_blunder_move(self, board: chess.Board, legal_moves: List[str]) -> Optional[chess.Move]:
        """Intentionally make a bad move (for lower difficulties)"""
        bad_moves = []
        for move_uci in legal_moves:
            move = chess.Move.from_uci(move_uci)
            # Consider moves that don't capture and don't give check as potential blunders
            if not board.is_capture(move):
                temp_board = board.copy()
                temp_board.push(move)
                if not temp_board.is_check():
                    bad_moves.append(move)
        
        if bad_moves:
            return np.random.choice(bad_moves)
        return None

# Initialize the chess engine
try:
    # Load your trained model (update paths as needed)
    chess_engine = ChessAIEngine(
        model_path="ULTRA_OPTIMIZED_CHESS_AI.h5",
        move_dict_path="ULTRA_OPTIMIZED_MOVE_DICT.json"
    )
    print("✅ Chess AI Engine loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load chess engine: {e}")
    # Fallback to random moves if model not available
    chess_engine = None

# Default session for demo purposes
DEFAULT_SESSION = "demo_session"
if chess_engine:
    chess_engine.game_states[DEFAULT_SESSION] = chess.Board()

# API Key verification middleware
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# API Routes
@app.get("/")
async def root():
    return {"message": "Chess AI API", "status": "running", "version": "1.0.0"}

@app.post("/move", response_model=MoveResponse)
async def make_move(request: MoveRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    
    if not chess_engine:
        raise HTTPException(status_code=500, detail="Chess engine not available")
    
    board = chess_engine.game_states.get(DEFAULT_SESSION)
    if not board:
        raise HTTPException(status_code=404, detail="Game session not found")
    
    # Validate and make player move
    try:
        move = chess.Move.from_uci(request.move)
        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")
        
        board.push(move)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid move format")
    
    # Check game status after player move
    if board.is_game_over():
        return MoveResponse(
            next_move=None,
            game_over=True,
            result=board.result(),
            fen=board.fen(),
            message="Game over after player move"
        )
    
    # Get AI response
    ai_move_uci = chess_engine.get_ai_move(board, request.difficulty)
    
    if ai_move_uci:
        try:
            ai_move = chess.Move.from_uci(ai_move_uci)
            if ai_move in board.legal_moves:
                board.push(ai_move)
            else:
                # Fallback to first legal move
                ai_move = list(board.legal_moves)[0]
                board.push(ai_move)
                ai_move_uci = ai_move.uci()
        except ValueError:
            # Fallback to first legal move
            ai_move = list(board.legal_moves)[0]
            board.push(ai_move)
            ai_move_uci = ai_move.uci()
    else:
        # No legal moves for AI (shouldn't happen in normal chess)
        ai_move_uci = None
    
    # Check game status after AI move
    game_over = board.is_game_over()
    result = board.result() if game_over else None
    
    return MoveResponse(
        next_move=ai_move_uci,
        game_over=game_over,
        result=result,
        fen=board.fen(),
        message="Move processed successfully"
    )

@app.post("/difficulty", response_model=DifficultyResponse)
async def set_difficulty(request: DifficultyRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    
    difficulty = request.difficulty
    elo_estimate = difficulty  # In a real system, this would be more sophisticated
    
    return DifficultyResponse(
        message=f"Difficulty set to {difficulty} ELO",
        current_difficulty=difficulty,
        elo_estimate=elo_estimate
    )

@app.post("/reset")
async def reset_game(request: ResetRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    
    if not chess_engine:
        raise HTTPException(status_code=500, detail="Chess engine not available")
    
    chess_engine.game_states[DEFAULT_SESSION] = chess.Board()
    
    return {
        "message": "Game reset successfully",
        "difficulty": request.difficulty,
        "fen": chess_engine.game_states[DEFAULT_SESSION].fen()
    }

@app.post("/undo")
async def undo_move(request: UndoRequest, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    
    if not chess_engine:
        raise HTTPException(status_code=500, detail="Chess engine not available")
    
    board = chess_engine.game_states.get(DEFAULT_SESSION)
    if not board:
        raise HTTPException(status_code=404, detail="Game session not found")
    
    # Undo last two moves (player and AI)
    moves_undone = 0
    if len(board.move_stack) >= 2:
        board.pop()  # AI move
        board.pop()  # Player move
        moves_undone = 2
    elif len(board.move_stack) >= 1:
        board.pop()  # Player move only
        moves_undone = 1
    
    return {
        "message": f"Undid {moves_undone} moves",
        "fen": board.fen(),
        "moves_remaining": len(board.move_stack)
    }

@app.get("/board", response_model=BoardResponse)
async def get_board(x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    
    if not chess_engine:
        raise HTTPException(status_code=500, detail="Chess engine not available")
    
    board = chess_engine.game_states.get(DEFAULT_SESSION)
    if not board:
        raise HTTPException(status_code=404, detail="Game session not found")
    
    return BoardResponse(
        board_fen=board.fen(),
        current_turn="white" if board.turn else "black",
        game_over=board.is_game_over(),
        result=board.result() if board.is_game_over() else None
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_loaded": chess_engine is not None,
        "active_sessions": len(chess_engine.game_states) if chess_engine else 0
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
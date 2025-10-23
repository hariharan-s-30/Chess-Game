import os
import chess
import chess.pgn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

# =========================
# 🔹 1. Encode Board
# =========================
def encode_board(board: chess.Board):
    mapping = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
        chess.QUEEN: 4, chess.KING: 5
    }
    enc = np.zeros(64 * 12, dtype=np.float32)
    for square, piece in board.piece_map().items():
        idx = mapping[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        enc[square * 12 + idx] = 1
    return enc


# =========================
# 🔹 2. Move Space (skip invalid self-moves)
# =========================
ALL_MOVES = []
for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        if from_sq == to_sq:
            continue  # skip a2a2 etc.
        move = chess.Move(from_sq, to_sq)
        ALL_MOVES.append(move.uci())
ALL_MOVES = sorted(set(ALL_MOVES))
MOVE2IDX = {m: i for i, m in enumerate(ALL_MOVES)}
IDX2MOVE = {i: m for m, i in MOVE2IDX.items()}


# =========================
# 🔹 3. Random PGN Generator
# =========================
def generate_random_game(max_moves=60):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    for _ in range(max_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
        node = node.add_variation(move)
    game.headers["Event"] = "Random Training Game"
    game.headers["Result"] = board.result()
    return game


def generate_pgn_file(filename="data.pgn", num_games=1000):
    print(f"⚙️ Generating {num_games} random games...")
    with open(filename, "w") as f:
        for i in range(num_games):
            game = generate_random_game()
            print(game, file=f, end="\n\n")
            if (i + 1) % 100 == 0:
                print(f"✅ Generated {i + 1} games")
    print("🎉 data.pgn ready!")


# =========================
# 🔹 4. Load PGN Dataset
# =========================
def load_dataset(pgn_path, max_games=None):
    X, y = [], []
    total_moves = 0
    games_used = 0

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if max_games is not None and games_used >= max_games:
                break
            board = game.board()
            moves_in_game = 0
            for move in game.mainline_moves():
                X.append(encode_board(board))
                y.append(MOVE2IDX.get(move.uci(), 0))
                board.push(move)
                moves_in_game += 1
            total_moves += moves_in_game
            games_used += 1

    print(f"✅ Games loaded: {games_used}")
    print(f"✅ Total moves extracted: {total_moves}")
    return np.array(X), np.array(y), games_used, total_moves


# =========================
# 🔹 5. Ensure All Moves Are Covered
# =========================
def ensure_all_moves_in_dataset(X, y):
    added = 0
    for move_uci, idx in MOVE2IDX.items():
        dummy_board = chess.Board()
        move_obj = chess.Move.from_uci(move_uci)
        if move_obj in dummy_board.legal_moves:
            X.append(encode_board(dummy_board))
            y.append(idx)
            added += 1
    print(f"✅ Added {added} synthetic samples to ensure all moves are covered")
    return X, y


# =========================
# 🔹 6. Build Model
# =========================
def build_model(move_count):
    model = keras.Sequential([
        layers.Input(shape=(64 * 12,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(move_count, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# =========================
# 🔹 7. Live Plot Callback (auto-save at end)
# =========================
class LivePlotCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        clear_output(wait=True)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, label="Train Loss")
        plt.plot(self.epochs, self.val_losses, label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.acc, label="Train Accuracy")
        plt.plot(self.epochs, self.val_acc, label="Val Accuracy")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def on_train_end(self, logs=None):
        # Save the final graph automatically
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, label="Train Loss")
        plt.plot(self.epochs, self.val_losses, label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.acc, label="Train Accuracy")
        plt.plot(self.epochs, self.val_acc, label="Val Accuracy")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_plot.png")
        print("📊 Training graph saved as 'training_plot.png'")


# =========================
# 🔹 8. Main Training
# =========================
if __name__ == "__main__":
    # Auto-generate PGN if missing
    if not os.path.exists("data.pgn"):
        print("⚠️ No data.pgn found. Creating one automatically...")
        generate_pgn_file("data.pgn", num_games=1000)

    # User inputs
    NUM_GAMES_TO_TRAIN = int(input("Enter the number of games to train on: "))
    NUM_EPOCHS = int(input("Enter the number of training epochs: "))

    # Load dataset
    X, y, games_used, total_moves = load_dataset("data.pgn", max_games=NUM_GAMES_TO_TRAIN)

    # Ensure every possible move appears
    X, y = list(X), list(y)
    X, y = ensure_all_moves_in_dataset(X, y)
    X, y = np.array(X), np.array(y)
    print("Final dataset shape:", X.shape, y.shape)

    # Build model
    model = build_model(len(MOVE2IDX))

    # Callbacks: Early stopping + Live plot
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    live_plot = LivePlotCallback()

    # Train model
    history = model.fit(
        X, y,
        epochs=NUM_EPOCHS,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, live_plot]
    )

    # Save model + metadata
    model.save("chess_ai_tf.keras")  # ✅ modern format
    pd.Series(MOVE2IDX).to_json("move_dict.json")

    stats = {
        "games_used": games_used,
        "total_moves": total_moves,
        "epochs": len(history.history['loss'])
    }
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)

    print("\n✅ Training complete!")
    print("✅ Model saved as chess_ai_tf.keras")
    print("✅ Move dictionary & stats saved")
    print("✅ Training graph saved as training_plot.png")

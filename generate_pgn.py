import chess
import chess.pgn
import random

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
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2025.10.04"
    game.headers["Round"] = "1"
    game.headers["White"] = "RandomWhite"
    game.headers["Black"] = "RandomBlack"
    game.headers["Result"] = board.result()

    return game

def generate_pgn_file(filename="data.pgn", num_games=1000):
    with open(filename, "w") as f:
        for i in range(num_games):
            game = generate_random_game()
            print(game, file=f, end="\n\n")
            if (i+1) % 100 == 0:
                print(f" Generated {i+1} games")

if __name__ == "__main__":
    generate_pgn_file("data.pgn", num_games=1000)
    print(" data.pgn with 1000 random games generated successfully")

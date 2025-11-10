# Tic Tac Toe Game (2 Players)

board = [" " for _ in range(9)]

def print_board():
    print("\n")
    for i in range(3):
        print(" | ".join(board[i*3:(i+1)*3]))
        if i < 2:
            print("--+---+--")
    print("\n")

def check_winner(player):
    win_patterns = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for pattern in win_patterns:
        if all(board[i] == player for i in pattern):
            return True
    return False

def is_full():
    return all(cell != " " for cell in board)

current_player = "X"

while True:
    print_board()
    move = int(input(f"Player {current_player}, enter position (1-9): ")) - 1
    if 0 <= move < 9 and board[move] == " ":
        board[move] = current_player
        if check_winner(current_player):
            print_board()
            print(f"🎉 Player {current_player} wins!")
            break
        elif is_full():
            print_board()
            print("It's a draw!")
            break
        current_player = "O" if current_player == "X" else "X"
    else:
        print("Invalid move. Try again.")

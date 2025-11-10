# 8 Puzzle Game (manual play version)

def print_board(board):
    for row in board:
        print(" ".join(str(x) if x != 0 else " " for x in row))
    print()

def find_empty(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j

def move(board, direction):
    i, j = find_empty(board)
    if direction == "up" and i > 0:
        board[i][j], board[i-1][j] = board[i-1][j], board[i][j]
    elif direction == "down" and i < 2:
        board[i][j], board[i+1][j] = board[i+1][j], board[i][j]
    elif direction == "left" and j > 0:
        board[i][j], board[i][j-1] = board[i][j-1], board[i][j]
    elif direction == "right" and j < 2:
        board[i][j], board[i][j+1] = board[i][j+1], board[i][j]
    else:
        print("Invalid move!")

def is_solved(board):
    return board == [[1,2,3],[4,5,6],[7,8,0]]

# Initial puzzle configuration
board = [[1,2,3],[4,0,6],[7,5,8]]

print("8 Puzzle Game\nUse commands: up, down, left, right")
while True:
    print_board(board)
    if is_solved(board):
        print("🎉 Congratulations! You solved the puzzle!")
        break
    move_input = input("Move: ").lower()
    move(board, move_input)

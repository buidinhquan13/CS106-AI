import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)    
    frontier = collections.deque([[startState]])    
    exploredSet = set() 
    actions = [[0]] 
    temp = []
    while frontier: 
        node = frontier.pop()   
        node_action = actions.pop() 
        if isEndState(node[-1][-1]): 
            temp += node_action[1:] 
            break
        if node[-1] not in exploredSet: 
            exploredSet.add(node[-1])   
            for action in legalActions(node[-1][0], node[-1][1]): 
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp 

def breadthFirstSearch(gameState):                  # Hàm nhận tham số gameState là trạng thái khởi đầu của trò chơi
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)                # Vị trí ban đầu của các BOX trong bản đồ
    beginPlayer = PosOfPlayer(gameState)            # Vị trí ban đầu của Player trong bản đồ
    
    startState = (beginPlayer, beginBox)            # Trạng thái bắt đầu gồm: vị trí bắt đầu của Player và các Box
    frontier = collections.deque([[startState]])    # store states, vị trí đầu tiên là vị trí bắt đầu của Player
    exploredSet = set()                             # Dùng set() để lưu các hành động mà Player đã đi qua và tránh trùng lặp
    actions = collections.deque([[0]])              # store actions, tạo hàng đợi queue actions (vị trí ban đầu là trạng thái 0)
    temp = []                                       # Lưu danh sách các hành động từ trạng thái bắt đầu đến trạng thái kết thúc
    while frontier:                                 # Kiểm tra frontier có rỗng hay không? Nếu không rỗng thì vào vòng lặp để xử lý
        node = frontier.popleft()                   # Lấy ra phần tử đầu tiên bên trái của frontier và xóa nó.
        node_action = actions.popleft()             # Lấy ra hành động đầu tiên bên trái của actions và xóa nó.
        if isEndState(node[-1][-1]):                # Kiểm tra xem đây có phải trạng thái kết thúc hay không?
            temp += node_action[1:]                 # Nếu là trạng thái kết thúc, thì lưu dãy hành động vào danh sách temp 
            break                                   # Thu được kết quả bài toán ==> kết thúc vòng lặp
        if node[-1] not in exploredSet:             # Kiểm tra node[-1] có nằm trong exploredSet hay không? (Player đã đi qua hay chưa?)
            exploredSet.add(node[-1])               # Nếu không có thì thêm node[-1] vào exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Xét tất cả các hành động hợp lệ có thể thực hiện từ hành động hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật lại vị trí Player và vị trí Box sau hành hộng hợp lệ đó.
                if isFailed(newPosBox):             # Kiểm tra vị trí mới của Box có hợp lệ hay không?
                    continue                        # Nếu Box ở vị trí không hợp lệ, bỏ qua và chuyển sang action khác.
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Thêm trạng thái mới vào frontier
                actions.append(node_action + [action[-1]])          # Thêm hành động mới vào actions dựa vào đường đi trước đó
    return temp                                     # Trả về danh sách các hành động đến trạng thái mục tiêu của thuật toán BFS

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()]) # Tính chi phí của hành động ứng với chữ viết thường
                                                    # Viết hoa: là Player đẩy vào thùng
                                                    # Viết thường: là Player di chuyển không chạm thùng

def uniformCostSearch(gameState):                   # Hàm nhận tham số gameState là trạng thái khởi đầu của trò chơi
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState)                # Vị trí ban đầu của các BOX trong bản đồ
    beginPlayer = PosOfPlayer(gameState)            # Vị trí ban đầu của Player trong bản đồ

    startState = (beginPlayer, beginBox)            # Trạng thái bắt đầu gồm: vị trí bắt đầu của Player và các Box
    frontier = PriorityQueue()                      # Hàng đợi Priority Queue frontier để lưu trữ các trạng thái của trò chơi, sắp xếp theo chi phí.
    frontier.push([startState], 0)                  # Thêm startState vào "frontier" với chi phí là 0
    exploredSet = set()                             # Dùng set() để lưu các hành động mà Player đã đi qua và tránh trùng lặp
    actions = PriorityQueue()                       # Hàng đợi Priority queue actions để lưu trữ các hành động, sắp xếp theo chi phí.
    actions.push([0], 0)                            # Thêm hành động ban đầu vào "actions" với chi phí là 0
    temp = []                                       # Lưu danh sách các hành động từ trạng thái bắt đầu đến trạng thái kết thúc
    while frontier:                                 # Kiểm tra frontier có rỗng hay không? Nếu không rỗng thì vào vòng lặp để xử lý
        node = frontier.pop()                       # Lấy ra phần tử đầu tiên bên trái của priority queue frontier và xóa nó.
        node_action = actions.pop()                 # Lấy ra hành động đầu tiên bên trái của Priority Queue actions và xóa nó.
        if isEndState(node[-1][-1]):                # Kiểm tra xem đây có phải trạng thái kết thúc hay không?
            temp += node_action[1:]                 # Nếu là trạng thái kết thúc, thì lưu dãy hành động vào danh sách temp 
            break                                   # Thu được kết quả bài toán ==> kết thúc vòng lặp
        if node[-1] not in exploredSet:             # Kiểm tra node[-1] có nằm trong exploredSet hay không? (Player đã đi qua hay chưa?)
            exploredSet.add(node[-1])               # Nếu không có thì thêm node[-1] vào exploredSet
            Cost = cost(node_action[1:])            # Tính chi phí hiện tại dựa trên số hành động đã thực hiện.
            for action in legalActions(node[-1][0], node[-1][1]): # Xét tất cả các hành động hợp lệ có thể thực hiện từ hành động hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Cập nhật lại vị trí Player và vị trí Box sau hành hộng hợp lệ đó.
                if isFailed(newPosBox):             # Kiểm tra vị trí mới của Box có hợp lệ hay không?
                    continue                        # Nếu Box ở vị trí không hợp lệ, bỏ qua và chuyển sang action khác.
                frontier.push(node + [(newPosPlayer, newPosBox)],Cost)  # Thêm trạng thái mới vào frontier với chi phí tương ứng. 
                                                                        # Chi phí được tính dựa trên số hành động đã thực hiện từ trạng thái ban đầu đến trạng thái hiện tại.
                actions.push(node_action + [action[-1]],Cost)           # Thêm hành động mới vào actions với chi phí tương ứng
    return temp                                     # Trả về danh sách các hành động đến trạng thái mục tiêu của thuật toán UCS
    
"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    print(len(result))
    return result

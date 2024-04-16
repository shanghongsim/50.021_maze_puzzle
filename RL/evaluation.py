from tqdm.notebook import tqdm
from environment import MazeEnvironment
from agent import Agent
import json
import ast
from sklearn.metrics import f1_score, accuracy_score

def evaluation(test_data, net, memory_buffer, test_config):
    overall_accuracy = []
    overall_f1 = []
    overall_num_moves_goal = []
    overall_solved_maze = []

    num_test_maze = len(test_data)

    for i in tqdm(range(num_test_maze)):
        solved_maze = True

        # get test maze
        test_maze = test_data[i]['maze'].numpy()

        # erase agent position in maze
        test_maze[0,0] = 0

        # maze env
        initial_position = [0,0]
        goal = [len(test_maze)-1,len(test_maze)-1]
        test_maze_env = MazeEnvironment(test_maze, initial_position, goal)

        # test agent
        test_agent = Agent(maze = test_maze_env,
                      memory_buffer = memory_buffer,
                      use_softmax = False
                    )
        net.eval()

        prediction = test_agent.predict(net)
        solution = json.loads(test_data[i]['solution'])

        num_correct = 0
        total = 0
        y_pred = []
        y_true = []

        for k in list(solution.keys()):
            pred_k = ast.literal_eval(k)
            pred_k = (pred_k[0]-1, pred_k[1]-1)

            if pred_k == (len(test_maze)-1, len(test_maze)-1) or k == f'({len(test_maze)}, {len(test_maze)})':
                continue
            
            gt_action = test_config.test_directions[solution[k]]
            pred_action = prediction[pred_k]

            num_correct += gt_action == pred_action
            total += 1

            if not (gt_action == pred_action) and solved_maze: # record first wrong move
                solved_maze = False
                overall_solved_maze.append(solved_maze)
                overall_num_moves_goal.append(test_config.max_step)
            
            y_true.append(gt_action)
            y_pred.append(pred_action)

        if solved_maze:
            overall_solved_maze.append(solved_maze)
            overall_num_moves_goal.append(total)

        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        # print(f1, accuracy)
        overall_accuracy.append(accuracy)
        overall_f1.append(f1)

    # print(overall_accuracy)
    # print(overall_f1)
    return overall_accuracy, overall_f1, overall_num_moves_goal, overall_solved_maze
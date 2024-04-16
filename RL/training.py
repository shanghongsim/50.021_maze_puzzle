from loss import Qloss
import torch
from environment import MazeEnvironment
from agent import Agent
from IPython.display import display, clear_output
import numpy as np

def train(agent, net, optimizer, epsilon, training_data, config):
    loss_log = []
    maze_change_log = []
    move_log = []
    result_log = []

    best_loss = 1e5
    running_loss = 0

    maze_idx = 0

    for epoch in range(config.num_epochs):
        loss = 0
        counter = 0
        eps = epsilon[epoch]

        agent.isgameon = True
        _ = agent.env.reset(eps)

        while agent.isgameon:
            agent.make_a_move(net, eps)
            counter += 1

            if len(agent.buffer) < config.buffer_start_size:
                continue

            optimizer.zero_grad()
            batch = agent.buffer.sample(config.batch_size, device = config.device)
            loss_t = Qloss(batch, net, gamma = config.gamma, device = config.device)
            loss_t.backward()
            optimizer.step()

            loss += loss_t.item()

        if (agent.env.current_position == agent.env.goal).all():
            result = 'won'
        else:
            result = 'lost'

        if epoch%1000 == 0:
            print("policy map")
            agent.plot_policy_map(net, f'{config.folder}/sol_epoch_{str(epoch)}.pdf', [0.35,-0.3])

        loss_log.append(loss)
        move_log.append(counter)
        result_log.append(result)

        if (epoch > 2000):
            running_loss = np.mean(loss_log[-50:])
            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(net.state_dict(), f'{config.folder}/best.torch')
                estop = epoch

        print('Epoch', epoch, '(number of moves ' + str(counter) + ')')
        print('Game', result)
        print('[' + '#'*(100-int(100*(1 - epoch/config.num_epochs))) +
              ' '*int(100*(1 - epoch/config.num_epochs)) + ']')
        print('\t Average loss: ' + f'{loss:.5f}')
        if (epoch > 2000):
            print('\t Best average loss of the last 50 epochs: ' + f'{best_loss:.5f}' + ', achieved at epoch', estop)

        if config.generalized:
            if np.mean(loss_log[-3:]) < 0.01 and result == 'won':
                maze_idx += 1
                new_maze= training_data[maze_idx]['maze'].numpy()

                initial_position = [0,0]
                goal = [len(agent.env.maze)-1, len(agent.env.maze)-1]
                new_maze_env = MazeEnvironment(new_maze, initial_position, goal)
                agent.change_maze(new_maze_env)
                print("Maze changed!")
                maze_change_log.append(epoch)

        clear_output(wait = True)
    return loss_log, maze_change_log, move_log, result_log
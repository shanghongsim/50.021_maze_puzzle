import torch
import torch.nn as nn

def Qloss(batch, net, gamma=0.99, device="cuda"):
    states, actions, next_states, rewards, _ = batch
    lbatch = len(states)
    state_action_values = net(states.view(lbatch,-1))
    state_action_values = state_action_values.gather(1, actions.unsqueeze(-1))
    state_action_values = state_action_values.squeeze(-1)

    next_state_values = net(next_states.view(lbatch, -1))
    next_state_values = next_state_values.max(1)[0]

    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * gamma + rewards

    return nn.MSELoss()(state_action_values, expected_state_action_values)
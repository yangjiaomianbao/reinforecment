"""枚举合法动作的 Discrete SAC，采用研究方案附录 B.4 的温度约定。"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .ddqn import ActionNetwork, soft_update, training_main
from .utils import set_seed


def temperature_loss(log_alpha, entropies, action_counts, ratio):
    """H 低于目标时，梯度下降增大 alpha；不向策略传播温度损失梯度。"""
    target = ratio * action_counts.to(dtype=entropies.dtype).log()
    return (log_alpha.exp() * (entropies.detach() - target)).mean()


class SACAgent:
    def __init__(self, graphs, config, *, seed=0, device="cpu"):
        p = config["rl"]
        if p["gamma"] != 1:
            raise ValueError("主实验要求 gamma=1")
        if not 0 < p["target_entropy_ratio"] < 1 or not np.isfinite(p["initial_alpha"]) or p["initial_alpha"] <= 0:
            raise ValueError("target_entropy_ratio 必须在 (0,1) 内且 initial_alpha 必须为有限正数")
        self.graphs, self.config, self.seed = dict(graphs), deepcopy(config), seed
        self.rng = np.random.default_rng(set_seed(seed, "rl"))
        # 独立构造三个完整编码器，不能共享 Actor/Critic 参数。
        self.actor = ActionNetwork(config).to(device)
        self.critics = nn.ModuleList(ActionNetwork(config) for _ in range(2)).to(device)
        self.targets = deepcopy(self.critics).requires_grad_(False).eval()
        self.log_alpha = nn.Parameter(torch.tensor(np.log(p["initial_alpha"]), dtype=torch.float32, device=device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=p["learning_rate"])
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=p["learning_rate"])
                                  for critic in self.critics]
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=p["learning_rate"])
        self.environment_steps = self.gradient_updates = 0
        self.critic_updates = self.actor_updates = self.temperature_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def policy(self, state, graph=None):
        if state.done:
            raise ValueError("终止状态不计算策略分布")
        graph = self.graphs[state.graph_id] if graph is None else graph
        scores = self.actor(graph, state)
        ids = torch.tensor(np.flatnonzero(state.alive_mask), device=scores.device, dtype=torch.long)
        log_probabilities = F.log_softmax(scores[ids], dim=0)
        return ids, log_probabilities.exp(), log_probabilities

    def act(self, state, *, graph=None, explore=False):
        with torch.no_grad():
            ids, probabilities, _ = self.policy(state, graph)
        if not explore:
            return int(ids[probabilities.argmax()].item())
        p = probabilities.cpu().numpy().astype(float)
        p /= p.sum()
        return int(self.rng.choice(ids.cpu().numpy(), p=p))

    @torch.no_grad()
    def td_targets(self, transitions):
        values = []
        for transition in transitions:
            value = self.log_alpha.new_tensor(transition.reward)
            if not transition.done:
                graph, state = self.graphs[transition.graph_id], transition.next_state
                ids, probability, log_probability = self.policy(state, graph)
                q = torch.minimum(self.targets[0](graph, state)[ids], self.targets[1](graph, state)[ids])
                value = value + (probability * (q - self.alpha * log_probability)).sum()
            values.append(value)
        return torch.stack(values)

    def critic_step(self, transitions):
        targets = self.td_targets(transitions)
        losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            predictions = torch.stack([critic(self.graphs[t.graph_id], t.state)[t.action_edge_id]
                                       for t in transitions])
            loss = F.smooth_l1_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("SAC Critic loss 非有限值")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.critic_updates += 1
            losses.append(float(loss.detach()))
        return losses

    def actor_step(self, states):
        # 清除 Critic 步残留梯度，再将所有合法 Q 值视为常量。
        for optimizer in self.critic_optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        for state in states:
            graph = self.graphs[state.graph_id]
            ids, probability, log_probability = self.policy(state, graph)
            with torch.no_grad():
                q = torch.minimum(self.critics[0](graph, state)[ids], self.critics[1](graph, state)[ids])
            losses.append((probability * (self.alpha.detach() * log_probability - q)).sum())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("SAC Actor loss 非有限值")
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        self.actor_updates += 1
        return float(loss.detach())

    def temperature_step(self, states):
        entropies, counts = [], []
        with torch.no_grad():
            for state in states:
                ids, probability, log_probability = self.policy(state)
                entropies.append(-(probability * log_probability).sum())
                counts.append(len(ids))
        # 单动作批次完全跳过优化器步，避免 Adam 历史动量仍改变温度。
        if all(count == 1 for count in counts):
            return 0.0
        loss = temperature_loss(self.log_alpha, torch.stack(entropies),
                                torch.tensor(counts, device=self.log_alpha.device),
                                self.config["rl"]["target_entropy_ratio"])
        if not torch.isfinite(loss):
            raise RuntimeError("SAC 温度 loss 非有限值")
        self.alpha_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.alpha_optimizer.step()
        self.temperature_updates += 1
        return float(loss.detach())

    def update(self, transitions):
        states = [transition.state for transition in transitions]
        critic_losses = self.critic_step(transitions)
        actor_loss = self.actor_step(states)
        alpha_loss = self.temperature_step(states)
        for target, critic in zip(self.targets, self.critics):
            soft_update(target, critic, self.config["rl"]["target_tau"])
        self.gradient_updates += 1
        return {"critic_losses": critic_losses, "actor_loss": actor_loss,
                "temperature_loss": alpha_loss, "alpha": float(self.alpha.detach())}

    def inference_state(self):
        return {"actor": deepcopy(self.actor.state_dict()), "critics": deepcopy(self.critics.state_dict()),
                "log_alpha": self.log_alpha.detach().clone()}

    def load_inference_state(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.targets.load_state_dict(state["critics"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])


if __name__ == "__main__":
    training_main(SACAgent, "sac")

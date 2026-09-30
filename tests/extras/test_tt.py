# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from transformers.optimization import get_scheduler

from llamafactory.extras.tt import (
    TTAdamW,
    TTLRScheduler,
    is_torch_tt_available,
    materialize_adamw_state,
    unwrap_optimizer,
)


def _make_optimizer(learning_rate: float) -> "torch.optim.Optimizer":
    params = [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(2))]
    optimizer = torch.optim.AdamW([{"params": params[:1]}, {"params": params[1:]}], lr=learning_rate)
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(learning_rate, dtype=torch.float32)

    return optimizer


def test_tt_not_requested_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.delenv("USE_TT", raising=False)
    assert is_torch_tt_available() is False


def test_tt_scheduler_streams_learning_rates():
    optimizer = _make_optimizer(1e-3)
    shadow = torch.optim.SGD([{"params": [torch.zeros(1)], "lr": 1e-3}, {"params": [torch.zeros(1)], "lr": 1e-3}])
    shadow._opt_called = True
    inner = get_scheduler("linear", optimizer=shadow, num_warmup_steps=2, num_training_steps=6)
    reference_optimizer = torch.optim.SGD([torch.zeros(1)], lr=1e-3)
    reference_optimizer._opt_called = True
    reference = get_scheduler("linear", optimizer=reference_optimizer, num_warmup_steps=2, num_training_steps=6)
    scheduler = TTLRScheduler(optimizer, inner)
    for _ in range(6):
        expected = reference.get_last_lr()[0]
        assert scheduler.get_last_lr()[0] == pytest.approx(expected)
        for group in optimizer.param_groups:
            assert torch.is_tensor(group["lr"])
            assert group["lr"].item() == pytest.approx(expected, abs=1e-9)

        scheduler.step()
        reference.step()

    state = scheduler.state_dict()
    restored = TTLRScheduler(
        _make_optimizer(1e-3),
        get_scheduler("linear", optimizer=shadow, num_warmup_steps=2, num_training_steps=6),
    )
    restored.load_state_dict(state)
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert restored._step_count == scheduler._step_count


def test_materialize_adamw_state_allocates_once():
    optimizer = _make_optimizer(1e-3)
    assert materialize_adamw_state(optimizer) == 2
    assert materialize_adamw_state(optimizer) == 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            assert state["step"].dtype == torch.float32
            assert torch.equal(state["exp_avg"], torch.zeros_like(param))
            assert torch.equal(state["exp_avg_sq"], torch.zeros_like(param))


def test_unwrap_optimizer_follows_wrappers():
    optimizer = _make_optimizer(1e-3)

    class Wrapper:
        def __init__(self, inner):
            self.optimizer = inner

    assert unwrap_optimizer(Wrapper(Wrapper(optimizer))) is optimizer


def test_tt_adamw_matches_torch_adamw():
    torch.manual_seed(0)
    weights = torch.randn(6, 5)
    reference_param = torch.nn.Parameter(weights.clone())
    tt_param = torch.nn.Parameter(weights.clone())
    reference = torch.optim.AdamW([reference_param], lr=1e-2, weight_decay=0.1)
    optimizer = TTAdamW([tt_param], lr=torch.tensor(1e-2), weight_decay=0.1)
    assert materialize_adamw_state(optimizer) == 1
    for step in range(5):
        grad = torch.randn(6, 5)
        reference_param.grad = grad.clone()
        tt_param.grad = grad.clone()
        if step == 0:
            optimizer.param_groups[0]["lr"].fill_(0.0)
            reference.param_groups[0]["lr"] = 0.0
        else:
            optimizer.param_groups[0]["lr"].fill_(1e-2)
            reference.param_groups[0]["lr"] = 1e-2

        reference.step()
        optimizer.step()
        assert torch.isfinite(tt_param).all()
        torch.testing.assert_close(tt_param, reference_param, rtol=1e-5, atol=1e-6)

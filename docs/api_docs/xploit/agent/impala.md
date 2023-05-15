#


## IMPALA
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L78)
```python 
IMPALA(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int, lr: float = 0.0004, eps: float = 0.01,
   use_lstm: bool = False, ent_coef: float = 0.01, baseline_coef: float = 0.5,
   max_grad_norm: float = 40, discount: float = 0.99,
   network_init_method: str = 'identity'
)
```


---
Importance Weighted Actor-Learner Architecture (IMPALA).


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": action_space.shape, "n": action_space.n, "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **use_lstm** (bool) : Use LSTM in the policy network or not.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **baseline_coef** (float) : Weighting coefficient of baseline value loss.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **discount** (float) : Discount factor.
* **network_init_method** (str) : Network initialization method name.



**Returns**

IMPALA distance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L146)
```python
.train(
   training: bool = True
)
```

---
Set the train mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .integrate
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L159)
```python
.integrate(
   **kwargs
)
```

---
Integrate agent and other modules (encoder, reward, ...) together

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L183)
```python
.act(
   *kwargs
)
```

---
Sample actions based on observations.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L187)
```python
.update(
   actor_model: nn.Module, learner_model: nn.Module, batch: Dict,
   init_actor_states: Tuple[th.Tensor, ...], optimizer: th.optim.Optimizer,
   lr_scheduler: th.optim.lr_scheduler, lock = threading.Lock()
)
```

---
Update the learner model.


**Args**

* **actor_model** (NNMoudle) : Actor network.
* **learner_model** (NNMoudle) : Learner network.
* **batch** (Batch) : Batch samples.
* **init_actor_states** (List[Tensor]) : Initial states for LSTM.
* **optimizer** (th.optim.Optimizer) : Optimizer.
* **lr_scheduler** (th.optim.lr_scheduler) : Learning rate scheduler.
* **lock** (Lock) : Thread lock.


**Returns**

Training metrics.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L256)
```python
.save(
   path: Path
)
```

---
Save models.


**Args**

* **path** (Path) : Storage path.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/agent/impala.py\#L267)
```python
.load(
   path: str
)
```

---
Load initial parameters.


**Args**

* **path** (str) : Import path.


**Returns**

None.
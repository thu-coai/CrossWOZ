# Train RL Policies

In the task-oriented system, a common strategy of learning a reinforcement learning dialog policy offline is to build a user simulator and make simulated interactions between the policy and simulator.

## Build an Environment

In tatk, we provide the enviroment class for training an RL policy, and we regard the all the components except system policy as the environment, i.e.:

```python
from tatk.dialog_agent.env import Environment

simulator = PipelineAgent(nlu_usr, dst_usr, policy_usr, nlg_usr)
env = Environment(nlg_sys, simulator, nlu_sys, dst_sys)
```

## Collect dialog sessions with multi-processing

To sample dialog sessions in a distributed setting, each process contains an actor that acts in its own copy of the environment.

```python
# sample data asynchronously
batch = sample(env, policy, batchsz, process_num)
```

Then, it returns all the samples to the main process and update the policy.

```python
# data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
# batch.action: ([1, a_dim], [1, a_dim]...)
# batch.reward/ batch.mask: ([1], [1]...)
s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
batchsz_real = s.size(0)

policy.update(epoch, batchsz_real, s, a, r, mask)
```

## Run

Please refer to [example_train.py](https://github.com/thu-coai/tatk/blob/master/tutorials/Train_RL_Policies/example_train.py) for details.

```bash
$ python example_train.py
```

You can change the following arguments in [example_train.py](https://github.com/thu-coai/tatk/blob/master/tutorials/Train_RL_Policies/example_train.py),

```python
batchsz = 1024
epoch = 20
process_num = 8
```

or `config.json` of corresponding RL policy during the training.


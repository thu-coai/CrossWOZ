# PPO

A policy optimization method in policy based reinforcement learning that uses
multiple epochs of stochastic gradient ascent and a constant
clipping mechanism as the soft constraint to perform each policy update. We adapt PPO to the dialog policy.

## Train

Refer to example_train.py and *Train RL Policies* in the tutorial.

## Reference

```
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```
# Dialog Policy

In the pipeline task-oriented dialog framework, the dialog policy module
takes as input the dialog state, and chooses the system action bases on
it.

This directory contains the interface definition of dialog policy
module for both system side and user simulator side, as well as some
implementations under different sub-directories.

## Interface

The interfaces for dialog policy are defined in policy.Policy:

- **predict** takes as input agent state (often the state tracked by DST)
and outputs the next system action.

- **init_session** reset the model variables for a new dialog session.

### Defination
define meta-dialog-act (meta-da) as a single act: 

such as `Request-hotel-area-?`, `Inform-train-destination-...`

define dialog-act (da) as a vector consisting several meta-das:

such as `[meta-da1, meta-da2]`

for clarity, we use `meta-da` as output in the test report 


### NLU Failed Dialog Act

let `x = user_output_da`, `y = sys_input_da`

`x --(nlg)--> natural language ---(nlu)--->  y`

nlu fails when `y != x`

both x and y are vectors that consist of several meta-das

`for every u in x and v in y`, we consider `(u, v)` is a `nlu-fail-pair`

each `nlu-fail-pair` will only be counted once in a single task round, considering it may appear more than once


### Cycle Failed Dialog Act / Cycle Failed Rate

if a certain `da` appears more than once in one domain task, we consider that is a `cycle-da`

if a domain task fails, we consider the last `cycle-da` in this domain as a `cycle-failed-da` (which means it is the most likely to cause the failure)

let `x` be a certain `cycle-failed-da`, for every `meta-da u` in `x`, we collect `u` as output

there is at most one `cycle-failed-da` in a single domain round

`cycle-failed-rate = #(cycle-failed-da) / #(domain round)`

### Bad Inform Dialog Act

let x be a `sys_output_da` 

consider every `meta-da u` in `x`, if the `slot of u` doesn't match the `value of u`, we consider it as a `bad-inform-da`

### Reqt-Not-Inform Dialog Act

let x be a `sys_output_da`, consider every `meta-da u` in `x`

let `s =  all the request slots that is in the initial goal`

consider every `slot v` in `s`, if there isn't any `u` in any `x` that matches `v`, we consider `v` as a `reqt-not-inform-slot`, we can add intent and domain information to `v`  to get a `meta-da`(just for unity)

### Inform-Not-Reqt Dialog Act

let x be a `sys_output_da` 

let `s =  all the request slots that is in the initial goal`

consider every `meta-da u` in `x`, if the `slot of u` is not in `s`, we consider it as a `inform-not-reqt-da`



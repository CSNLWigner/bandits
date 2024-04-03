## Exponential model

### Simple exponential model

The exponential model fits the behaviour by calculating an action value based on the outcomes of the previous 8 trials, that will lead to a preadiction of the probability of stay.   


**Action Value** = $\sum\limits_{i=0}^8 a*e^{-b * t}+c$

**Probability of stay** = $\sigma [\sum\limits_{i=0}^8 a*e^{-b * t}+c ]$

Here we fit the parameters a, b, and c. 

More complex models defined as strategy aware, and strategy unaware models, by distinguishing if $target_{t-1} == target_{t0}$. Further model versions were tried by introducing a2 and epsilon, as free parameters. 

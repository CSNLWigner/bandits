## Exponential model

### Simple exponential model

The exponential model fits the behaviour by calculating an action value based on the outcomes of the previous 8 trials, that will lead to a preadiction of the probability of stay.   


**Action Value** = $\sum\limits_{i=0}^8 a*e^{-b * t}+c$

**Probability of stay** = $\sigma [\sum\limits_{i=0}^8 a*e^{-b * t}+c ]$

Here we fit the parameters a, b, and c. 

More complex models defined as strategy aware, and strategy unaware models, by distinguishing if $target_{t-1} == target_{t0}$. Further model versions were tried by introducing a2 and epsilon, as free parameters. See the model descriptions <a href="https://www.notion.so/dbad05b926a34dddbd0f5f1d99221de9?v=67a739aab2bf4bf8b427e4a878c4b5e7">here</a>.


### CODES 
**Model fitting:** 

- exponential_model.py - this file contains the exponential model. 

- EXPONENTIAL MODEL CATALOGE - for monkey Kate - model fits, log losses for Kate

- EXPONENTIAL MODEL CATALOGE - for monkey Popey - model fits, log losses for Popey


## Logistic model 
**Action Value** = $\beta_{1} + \beta_{2} + \beta_{3} .. \beta_{8}$

- $p_{(swithc)} = \frac{e^{\beta{0}+\beta{1}*R_{t-1}+\beta{2}*R_{t-2}+...+\beta_{8}*R_{t-8}}}{1 + e^{\beta{0}+\beta{1}*R_{t-1}+\beta{2}*R_{t-2}+...+\beta_{8}*R_{t-8}}} = \sigma (\beta{0}+\beta{1}*R_{t-1}+\beta{2}*R_{t-2}+...+\beta_{8}*R_{t-8})  = \sigma(V_{t})$

### CODES 

- Logistic regression vs. empirical switch probability




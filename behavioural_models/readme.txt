** Exponential model **

The exponential model fits the behaviour by approximating the action value based on the outcomes of the previous 8 trials.  


Action Value = $ \sum _{t = 1} ^{8} \beta(t) = \sum _{t = 1} ^{8} a*e^{-b*t}+c$

## Probability of stay = $ \sigma  [(\sum _{t=1} ^{8} a*e^{b*t}*r_{t})+c]$

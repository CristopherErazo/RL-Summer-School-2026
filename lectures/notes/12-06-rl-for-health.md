# RL for Health — Kelly Zhang, 12/06/26

## Summary

**Challenges in Health care**

The states are not fully observed, we get an observation which is either a corrupt version of the state or an incomplete version. Given the incomplete information the process is not Markovian anymore.

**Incomplete state -> Confounding in online RL**

To mitigate the effect of confounding:

* Causal RL: Try to learn the causal graph, instrumental variables
* Sensitivity Analysis: How much confounding would it need to be to reverse the findings.
* Negative Controls: outcomes that are known to not be afferted by a treatment -> ensuring no causal effect is found.


**Exterbak Validity**

Data from different hospitals carry a massive distribution shift. 

## Papers & References
-

## Questions I Still Have
-


# Robotics — Georgia Chalvatzaki, 12/06/26

## Summary

The theory and algorithms or RL are more or less 'solved', now we need to understand how to connect that with the embodiment of algorithms into robots. 

Robots are not 'gym' environments.

* In gym loop: policy -- action --> simulator (full state, deterministic, reset) -- state, reward --> millions of cheep steps.
* The robot loop: policy -- torque/target --> robot+world (contact, wear, real time) -- noisy partial obs --> seconds each, human reset. 

* $s$: images, proprioception, force-torque: partial and delayed.
* $a$: a command to a controller, at some level of a hierarchy.
* $r$: you design it, and on contact tasks that is most of the work.




## Papers & References
-

## Questions I Still Have
-


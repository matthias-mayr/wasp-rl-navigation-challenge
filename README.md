# WASP AI Summer School RL Challenge
Solution for the WASP AI 2020 Summer School Reinforcement Learning navigation challenge in Minecraft.


# Benchmarking
* Environment 'MineRLNavigate-v0'
* 11 runs
* maximum 1000 steps
## With rendering the environment
```
Total time spent for steps:  49.63835620880127
Time per step:  0.009007141391544415
Total time to reset the environment:  239.53573989868164
Time per environment reset:  21.775976354425605
```
## Using a headless software renderer
```
Total time spent for steps:  93.57975935935974
Time per step:  0.017303949585680427
Total time to reset the environment:  242.27474236488342
Time per environment reset:  22.024976578625765
```


# Troubleshooting
## "Could not determine java version from '11.0.?'."
**Problem:**

There are build errors like
```
EOFError:
FAILURE: Build failed with an exception.

* What went wrong:
Could not determine java version from '11.0.8'.
```

**Solution:**

You are not using the right version of Java. On Linux execute:
```
scripts/switch_java.sh
```

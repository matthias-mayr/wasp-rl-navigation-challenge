# WASP AI Summer School RL Challenge
Solution for the WASP AI 2020 Summer School Reinforcement Learning navigation challenge in Minecraft.


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

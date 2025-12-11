Failed attempts:
Getting rid of CastLinear does not really save memory sadly.


1. Add checkpointing. Specifically in the `for _L_step in range(self.config.L_cycles):`  loop for TinyRecursiveReasoningModel_ACTV1_Inner's forward pass. This ends up cutting down peak cuda memory reserved from 41.31 GB to  7.88 GB on eager without a speed decrease. However, we need to validate loss curves using the two models. It also cuts down attn to 5.05 GB from 24.01. The issue is we are no longer able to use cuda graphs however normal torch.compile does work and gives us the same perf roughly.

2. Another quirk about this model is that there is a bunch of casting up to float32. Removing all of this reduces peak memory in eager with mlp further to 6.94. In eager perf goes for from ~2seconds to ~1.8 seconds for the 5 iterations. Peak memory after compile interestingly goes up to 8.33 GB from 7.88 GB. However, we now hit 744.565ms per step. This is the same as the baseline torch.compile.
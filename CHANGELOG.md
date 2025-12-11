Failed attempts:
Getting rid of CastLinear does not really save memory sadly.


1. Add checkpointing. Specifically in the `for _L_step in range(self.config.L_cycles):`  loop for TinyRecursiveReasoningModel_ACTV1_Inner's forward pass. This ends up cutting down peak cuda memory reserved from 41.31 GB to  7.88 GB on eager without a speed decrease. However, we need to validate loss curves using the two models. It also cuts down attn to 5.05 GB from 24.01.

2.
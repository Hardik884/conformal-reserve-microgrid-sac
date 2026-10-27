# Preliminary capstone notebooks (kept for the record)

These two Colab notebooks are the **earlier capstone study** that the paper grew out of. They are **not** used by the code in `../code/` and their numbers are **not** the paper's results.

- `microgrid_SAC_CQR_fixed.ipynb`: CNN-LSTM forecaster + CQR + SAC.
- `microgrid_SAC_no_CQR_ablation_v2.ipynb`: the same without CQR (ablation).

Why the paper replaces them: they used Pecan Street data (not redistributable), a simulator that allowed demand to go unserved, training and testing on overlapping data, and an improvement ratio computed against a zero baseline. The paper's headline "51% less grid import" came from this earlier setup and should not be cited. See Section 7 of the paper ("What changed from the preliminary study").

The Pecan Street data is **not** included. Cell outputs that printed raw data rows were cleared; plots and summary statistics were kept.

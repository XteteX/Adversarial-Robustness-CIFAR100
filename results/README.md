# Results

Numbers copied verbatim from the output of section 8 of the notebook
(`notebooks/adversarial_robustness_cifar100.ipynb`). All values are **mean and standard
deviation over 3 random seeds**; accuracies in %, Macro F1 in [0, 1].

| File | Content |
|---|---|
| `eps8_all_attacks.csv` | Clean accuracy, Macro F1, FGSM, PGD-7, PGD-20 and AutoAttack accuracy at ε = 8/255, plus NRR computed from AutoAttack (Table I and Table IV of the paper) |
| `pgd20_by_eps.csv` | Per-budget results (ε = 2, 4, 8 /255): clean, PGD-7 and PGD-20 accuracy, NRR, conditional robust accuracy ρ and the logit-MSE ratio r (Tables II and V) |

`checkpoint = final` is the last-epoch model (main result in the paper); `best` is the
checkpoint selected on the 1,000-image validation split (secondary result).
An empty `logit_mse_ratio` means fewer than 30 samples were defended, so the ratio is not reported.
AutoAttack was run on the first 1,000 test images.

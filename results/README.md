# Results

Numbers copied verbatim from the output of section 8 of the notebook
(`notebooks/adversarial_robustness_cifar100.ipynb`). All values are **mean and standard
deviation over 3 random seeds**; accuracies in %, Macro F1 in [0, 1].

| File | Content |
|---|---|
| `eps8_all_attacks.csv` | Clean accuracy, Macro F1, FGSM, PGD-7, PGD-20 and AutoAttack accuracy at ε = 8/255, plus NRR computed from AutoAttack (Table I and Table IV of the paper) |
| `pgd20_by_eps.csv` | Per-budget results (ε = 2, 4, 8 /255): clean, PGD-7 and PGD-20 accuracy, NRR, conditional robust accuracy ρ and the logit-MSE ratio r (Tables II and V); VGG-16-lin rows added in the revision (PGD-7 and NRR std not recorded) |
| `revision_autoattack_and_vgg16lin.csv` | All six configurations: clean, PGD-20 (full test set) and AutoAttack (first 1,000 images) at ε = 4/255 and 8/255 |
| `same_subset_pgd20_vs_autoattack.csv` | PGD-20 and AutoAttack on the same first 1,000 test images (gradient-masking check) |
| `gap_decomposition.csv` | Decomposition of the PGD-20 gap (Eq. 7) with std over seed pairs, both reference orders, for ResNet-18 vs VGG-16 and vs VGG-16-lin |

`checkpoint = final` is the last-epoch model (main result in the paper); `best` is the
checkpoint selected on the 1,000-image validation split (secondary result).
An empty `logit_mse_ratio` means fewer than 30 samples were defended, so the ratio is not reported.
AutoAttack was run on the first 1,000 test images.

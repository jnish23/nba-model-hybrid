# E28 exploratory follow-ups (NOT registered)

Written after the registered screen (`screen.md`) was read. They check two confounds the screen exposed and one practical question; they change no tier. Every number here is a forking-path number and would need its own registration before it could support a decision.

## X0. Reproducibility of the registered numbers

a|F1|total now: Wald p 0.020699, gamma 0.592209; registered run: 0.020699, 0.592209. Identical.

## X1. (a) model-residual tests with the pace of the previous game and the team pace norm as controls

F1 divides distance by possessions, so a slow previous game raises `xload`, `dist_per_poss_last` and `rot_xload`; teams repeat their pace, so the total (and the pace outcome) moves with the previous game's pace whatever the legs did. F3 carries the previous game's pace directly. If the family only matters through pace, the controls remove it.

| family | target | spec | n find | find Wald p | gamma | gamma 95% CI | one-sided p |
|---|---|---|---|---|---|---|---|
| F1 | total | registered (no controls) | 3354 | 0.0207 | 0.592 | [0.18, 0.97] | 0.004 |
| F1 | total | + previous-game pace and team pace norm | 3354 | 0.0109 | 0.123 | [-1.99, 1.98] | 0.4725 |
| F1 | margin | registered (no controls) | 3354 | 0.4854 | 0.348 | [-0.04, 0.72] | 0.037 |
| F1 | margin | + previous-game pace and team pace norm | 3354 | 0.6663 | 0.838 | [-0.26, 1.96] | 0.0639 |
| F3 | total | registered (no controls) | 3366 | 0.6625 | 0.341 | [-0.17, 0.82] | 0.0819 |
| F3 | total | + previous-game pace and team pace norm | 3366 | 0.6963 | 0.66 | [0.13, 1.16] | 0.013 |
| F3 | margin | registered (no controls) | 3366 | 0.1248 | 0.335 | [-0.02, 0.76] | 0.036 |
| F3 | margin | + previous-game pace and team pace norm | 3366 | 0.1368 | 0.726 | [0.08, 1.47] | 0.015 |

## X2. Mechanism (c) for F1 and F2 with pace controls

Controls: own and opponent previous-game pace and prior EWMA pace.

| family | outcome | spec | n find | find Wald p | gamma | gamma 95% CI | one-sided p | rule (find p<0.10, gamma>0, p<0.0167) |
|---|---|---|---|---|---|---|---|---|
| F1 | 3P% | registered | 21120 | 0.1886 | 0.306 | [0.11, 0.49] | 0.001 | False |
| F1 | 3P% | + pace controls | 21120 | 0.5345 | 0.1 | [-0.55, 0.86] | 0.3586 | False |
| F1 | FT% | registered | 21120 | 0.0 | 0.486 | [0.22, 0.76] | 0.001 | True |
| F1 | FT% | + pace controls | 21120 | 0.2464 | 0.634 | [-0.39, 1.58] | 0.1119 | False |
| F1 | pace | registered | 21120 | 0.0 | 1.652 | [1.55, 1.76] | 0.001 | True |
| F1 | pace | + pace controls | 21120 | 0.005 | -0.087 | [-0.18, 0.01] | 0.963 | False |
| F2 | 3P% | registered | 21148 | 0.9658 | -0.03 | [-0.19, 0.12] | 0.6803 | False |
| F2 | 3P% | + pace controls | 21148 | 0.9032 | 0.239 | [0.06, 0.42] | 0.004 | False |
| F2 | FT% | registered | 21148 | 0.1141 | 0.231 | [0.03, 0.43] | 0.013 | False |
| F2 | FT% | + pace controls | 21148 | 0.1672 | 0.289 | [0.07, 0.50] | 0.006 | False |
| F2 | pace | registered | 21148 | 0.0 | 0.264 | [0.18, 0.35] | 0.001 | True |
| F2 | pace | + pace controls | 21148 | 0.0593 | -0.011 | [-0.10, 0.08] | 0.6204 | False |

## X3. The (b) signals applied to the closing line (confirmation seasons)

Same find-period signal (built from the opening-line residual), confirmation residual taken against the CLOSING line instead. A gamma near 0 against the close means the market had absorbed the information by the close; a bet on the opening number would then show positive closing-line value.

| family | target | opening gamma | opening 95% CI | n (closing) | closing gamma | closing 95% CI | closing one-sided p |
|---|---|---|---|---|---|---|---|
| F1 | margin | 0.485 | [0.12, 0.83] | 4767 | 0.448 | [0.08, 0.81] | 0.008 |
| F3 | margin | 0.538 | [0.15, 0.95] | 4885 | 0.495 | [0.11, 0.91] | 0.01 |
| F4 | margin | 0.42 | [0.03, 0.80] | 4893 | 0.257 | [-0.12, 0.68] | 0.0799 |
| F5 | total | 0.747 | [0.24, 1.21] | 4770 | 0.515 | [-0.00, 0.99] | 0.029 |
| F4 | total | 0.192 | [-0.36, 0.78] | 4896 | 0.026 | [-0.52, 0.55] | 0.4336 |
| F1 | total | 0.28 | [-0.20, 0.74] | 4770 | 0.168 | [-0.30, 0.63] | 0.2458 |

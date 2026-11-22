# Bergentrück Chess Engine

## Overview

Bergentruck (Bergentrück) is a Rust chess engine developed by ZackGabri and Tamplite Siphron Kents, the latter of whom developed the [Dragonrose chess engine](https://github.com/TampliteSK/dragonrose_cpp).

## Specifications

### Strength

|  Version  | CCRL 2+1 | COPE All-Round Bullet | COPE All-Round Rapid |
| :-------: | :------: | :-------------------: | :------------------: |
|v0.4 (dev) |  ~3269   |           -           |           -          |          
|   v0.3    |  ~3259   |      3133 +/- 11      |      3262 +/- 18     |
|   v0.2    |  ~2434   |           -           |           -          |
|   v0.1    |  ~1879   |           -           |           -          |

- CCRL 2+1 are based on STC progtests against various versions of Stash.
- COPE chess link: https://cope-chess.live/

### Board representation

Bergentruck uses [shakmaty crate](https://crates.io/crates/Shakmaty) for board representation and movegen, a library developed by Niklas Fiekas (niklasf) et al.

### Search

- Negamax with Alpha-beta Pruning (fail-soft)
    - General heuristics: Improving
    - Reverse futility pruning (RFP)
    - Null move pruning (NMP)
    - Futility pruning (FP) (commented)
    - Late move pruning (LMP)
    - Principal variation search (PVS)
    - Late move reductions (LMR)
- Quiescence search (fail-soft)
- Transposition table

- Move ordering: Hash move, MVV-LVA, quiet history moves, quiet moves
- Time management: Hard and soft time management

### Evaluation

#### NNUE
As of v0.3 and later versions, Bergentruck uses an NNUE network trained on millions of self-play games, played at 5000 soft nodes on a self-hosted OpenBench instance. The network is then trained on [Bullet](https://github.com/jw1912/bullet).
Inference code is largely based on example code given in Bullet as well, with minor optimisation done with AVX2 SIMD.

You will find a list of `.bin` networks under src/nn/networks/. Here is a table to illustrate the difference between the networks:
|  Network Family  |  Notable Net* |     Architecture     | Engine Version |
| :--------------: | :-----------: | :------------------: | :------------: |
|       Roulx      |   Roulx III   | (768 -> 1024)x2 -> 1 |      v0.3      |
|       King       |     King I    | (768 ->  512)x2 -> 1 |        -       |
|       Hathy      |    Hathy Ib   | (768 ->  256)x2 -> 1 |        -       |
|      Rudinn      |   Rudinn III  | (768 ->  128)x2 -> 1 |        -       |
|      Toriel      |   Toriel IV   | (768 ->   32)x2 -> 1 |        -       |

*Refers to either the strongest net in the family, or a net that was used in a given version release of the engine.

As the list grows bigger, weaker networks may be moved to another repository to reduce repo bloat. 
Yes, the networks are named after Deltarune characters. Very original, we know.

#### HCE
Older versions v0.1 and v0.2 use a hand-crafted evaluation (HCE) based on PesTO evaluation function. 
This legacy function is still present in the codebase.

#### Others?
In the future, there may be experiments with other neural net architectures or machine learning techniques. Stay tuned.

## Etymology

Bergentruck is a reference to Undertale / Deltarune, both games developed by Toby Fox et al. In Undertale, the soundtrack for Asgore's boss theme is Bergentrücking (German, lit. "King in the Mountain"), while in Deltarune there is a [popular meme](https://knowyourmeme.com/memes/asgore-running-over-dess) (animation + song) about Asgore running over December (Dess) Holiday with his truck. It is the combination of both of these references that lead to the culmination of the engine's name.

## Changelogs
v0.4 (dev): More search progression. <br>
v0.3: Add improving heuristics and fix bugs. Add NNUE and iterated on nets. <br>
v0.2: Bugfixes and added more search enhancements up to IIR and soft time management. <br>
v0.1: First release. Search progression up to NMP except aspiration windows.

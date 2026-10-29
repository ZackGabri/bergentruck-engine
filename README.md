# Bergentrück Chess Engine
## Overview
Bergentruck (Bergentrück) is a Rust chess engine developed by ZackGabri and Tamplite Siphron Kents, the latter of whom developed the Dragonrose chess engine.

## Specifications
### Strength 
| Version | CCRL 2+1 |
|:-------:|:--------:|
|  v0.3   |  ~3171   |
|  v0.2   |  ~2434   |
|  v0.1   |  ~1879   |

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
Currently using NNUE with a (768 -> 128)x2 -> 1 net trained on 1 million self-play games, played at 5000 soft nodes.
Minor optimisation done with AVX2 SIMD.
Trainer used and inference code adapted from: [Bullet](https://github.com/jw1912/bullet)
Legacy HCE evaluation code remains in the codebase.
Possibly other NN architectures / ML techniques in the future...

## Etymology
Bergentruck is a reference to Undertale / Deltarune, both games developed by Toby Fox et al. In Undertale, the soundtrack for Asgore's boss theme is Bergentrücking (German, lit. "King in the Mountain"), while in Deltarune there is a [popular meme](https://knowyourmeme.com/memes/asgore-running-over-dess) (animation + song) about Asgore running over December (Dess) Holiday with his truck. It is the combination of both of these references that lead to the culmination of the engine's name.

## Changelogs
v0.3 (dev): Add improving heuristics and fix bugs. Add NNUE and iterated on nets. <br>
v0.2: Bugfixes and added more search enhancements up to IIR and soft time management. <br>
v0.1: First release. Search progression up to NMP except aspiration windows, which failed to gain.
12345678
87654321
*****
Better evaluation function description:
    The idea behind this evaluation function is, as you suggested - combining some heuristics
    together in order to create a better one. All the heuristics are normalized to 1 in order to give them the same
    effect (and then we can multiply each one by a constant in order to give each one a different weight.
    The three heuristics are:
    1. "symmetric formation", as an individual who played tons of 2048 games in the past, the best strategy
    is to create a symmetric formation. Meaning - top left is the highest tile, and as we go away from it
    the values decrease. For example:
    64 32 16 0
    32 16 0  0
    16 0  0  0

    2. "free tiles" we want to have a lot of "free tile", so we give score for the amount of
    free tiles divided by the amount of possible free tiles.

    3. highest to the corner. The symetric formation is nice, but we want to keep the highest
    tile in one of the corners even if when it is not the formation is less symmetric. So, we gave
    "bonus" points for boards where the highest is in the corner.

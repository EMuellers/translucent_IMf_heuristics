def compute_f_1_scores(fitness_scores: list, precision_scores: list) -> list:
    i = 0
    result = []
    while i < len(fitness_scores):
        f = fitness_scores[i]
        p = precision_scores[i]
        if p == 0 or f == 0:
            result.append(0)
        else:
            f1 = 2 * (f * p) / (f + p)
            result.append(f1)
        i += 1
    return result

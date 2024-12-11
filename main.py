import sys
from collections import defaultdict, Counter

# Considered verified.
def load_sequence_and_confidence(seq_file, conf_file):
    """Load a sequence and corresponding confidence values."""
    with open(seq_file, 'r') as f:
        D = f.read().strip()

    with open(conf_file, 'r') as f:
        confidence_values = list(map(float, f.read().strip().split()))

    if len(D) != len(confidence_values):
        raise ValueError("Sequence length and confidence values length do not match.")

    return D, confidence_values

# Considered verified.
def load_substitution_matrix(matrix_file):
    """
    Load a substitution matrix from a file.
    Expected format:
        First line: A C G T (or chosen nucleotides)
        Following lines: symmetric matrix of scores
    """
    with open(matrix_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    header = lines[0].split()
    mat = {}
    for i, line in enumerate(lines[1:]):
        parts = line.split()
        row_nuc = header[i]
        for j, val in enumerate(parts):
            col_nuc = header[j]
            if row_nuc not in mat:
                mat[row_nuc] = {}
            mat[row_nuc][col_nuc] = float(val)
    return mat

# Considered verified.
def nucleotide_probabilities(p_max, max_nucleotide='A'):
    """Given p_max and its nucleotide, assign equal probability to other three nucleotides."""
    nucleotides = ['A', 'C', 'G', 'T']
    others = [n for n in nucleotides if n != max_nucleotide]
    remaining_prob = (1 - p_max)
    if remaining_prob < 0:
        remaining_prob = 0
    equal_share = remaining_prob / 3.0 if remaining_prob > 0 else 0.0
    probs = {max_nucleotide: p_max}
    for o in others:
        probs[o] = equal_share
    return probs

def compute_wmer_probability(wmer, wmer_probs):
    p = 1.0
    for i, nuc in enumerate(wmer):
        p *= wmer_probs[i].get(nuc, 0.0)
        if p == 0.0:
            break
    return p

# Mostly verified.
def generate_top_wmers_for_position(D, conf_values, i, w, max_candidates):
    if i + w > len(D):
        return [], []

    wmer_seq = D[i:i+w]
    wmer_probs = []
    p_max_list = []
    nucleotides = ['A', 'C', 'G', 'T']

    # Build probability distributions for each position in this w-mer
    for pos in range(i, i+w):
        max_nuc = D[pos]
        p_max = conf_values[pos]
        dist = nucleotide_probabilities(p_max, max_nuc)
        wmer_probs.append(dist)
        p_max_list.append((pos - i, dist[max_nuc])) 

    # Sort positions by ascending p_max_value
    p_max_list.sort(key=lambda x: x[1])

    # Start with the maximum-likelihood w-mer:
    best_wmer = "".join(D[i+j] for j in range(w))
    candidates = [best_wmer]

    def substitute_one_position(wmer_str, rel_pos):
        orig_nuc = wmer_str[rel_pos]
        variants = []
        for nuc in nucleotides:
            if nuc != orig_nuc:
                var = wmer_str[:rel_pos] + nuc + wmer_str[rel_pos+1:]
                variants.append(var)
        return variants

    rounds = len(p_max_list) # up to 3 positions to vary 
    for idx_in_sorted in range(rounds):
        rel_pos = p_max_list[idx_in_sorted][0]
        new_variants = []
        if best_wmer in candidates:
            new_variants.extend(substitute_one_position(best_wmer, rel_pos))
        candidates.extend(new_variants)
        if len(candidates) >= max_candidates:
            break

    candidates = candidates[:max_candidates]
    return candidates, wmer_probs

# Verified.
def build_wmer_map(D, conf_values, w, max_candidates, probability_threshold):
    f_map = defaultdict(list)
    for i in range(len(D) - w + 1):
        wmers, wmer_probs = generate_top_wmers_for_position(D, conf_values, i, w, max_candidates)
        filtered_wmers = []
        for wm in wmers:
            p = compute_wmer_probability(wm, wmer_probs)
            if p >= probability_threshold:
                filtered_wmers.append(wm)
        for wm in filtered_wmers:
            f_map[wm].append(i)
    return f_map

# Verified.
def find_seeds(q, f_map, w):
    seeds = []
    for start in range(len(q) - w + 1):
        q_wmer = q[start:start+w]
        if q_wmer in f_map:
            for pos in f_map[q_wmer]:
                seeds.append((q_wmer, start, pos))
    return seeds

# Unverified.
def expected_score(M, probs, q_nuc):
    score = 0.0
    for x in ['A', 'C', 'G', 'T']:
        score += probs[x] * M[q_nuc][x]
    return score

# Consider verified.
def compute_background_distribution(D, conf_values):
    """Compute average background probabilities over D."""
    total_positions = len(D)
    nucleotides = ['A','C','G','T']
    sum_probs = {n:0.0 for n in nucleotides}
    for i in range(total_positions):
        max_nuc = D[i]
        p_max = conf_values[i]
        dist = nucleotide_probabilities(p_max, max_nuc)
        for n in nucleotides:
            sum_probs[n] += dist[n]
    for n in nucleotides:
        sum_probs[n] /= total_positions
    return sum_probs

# Consider verified.
def compute_query_distribution(q):
    """Compute frequency distribution of nucleotides in q."""
    count = Counter(q)
    length = len(q)
    return {
        'A': count.get('A',0)/length,
        'C': count.get('C',0)/length,
        'G': count.get('G',0)/length,
        'T': count.get('T',0)/length
    }

# Unverified.
def compute_expected_score_overall(M, background_probs, query_probs):
    """
    E[score] = sum_{q_n,x} q_p(q_n)*bar{p}(x)*M(q_n,x)
    """
    expected = 0.0
    for q_n in ['A','C','G','T']:
        for x in ['A','C','G','T']:
            expected += query_probs[q_n]*background_probs[x]*M[q_n][x]
    return expected

def ungapped_extension(q, D, conf_values, seed_q_start, seed_d_start, w, M, dropoff_threshold, verbose=False):
    """
    Perform an ungapped extension from a seed to create an HSP.
    Extends left and right as long as the expected score increases.
    """
    # Initialize
    q_start = seed_q_start
    q_end = seed_q_start + w - 1
    d_start = seed_d_start
    d_end = seed_d_start + w - 1

    def get_probs(pos):
        max_nuc = D[pos]
        p_max = conf_values[pos]
        return nucleotide_probabilities(p_max, max_nuc)

    # Scores the seed using the expected score of each position
    current_score = 0.0
    for offset in range(w):
        q_nuc = q[q_start + offset]
        d_probs = get_probs(d_start + offset)
        s = expected_score(M, d_probs, q_nuc)
        current_score += s

    max_score = current_score
    if verbose:
        print(f"\nStarting ungapped extension for seed at Q:{q_start}-{q_end}, D:{d_start}-{d_end}")
        print(f"Initial ungapped score: {current_score:.2f}")

    # Extend to the left
    # left_extension = 1
    while q_start > 0 and d_start > 0: # Prevents it from going out the left boundary of q and D
        q_pos = q_start - 1
        d_pos = d_start - 1
        q_nuc = q[q_pos]
        d_probs = get_probs(d_pos)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s

        if verbose:
            print(f"Extending left to Q[{q_pos}]={q_nuc}, D[{d_pos}]")
            print(f"Expected score: {s:.2f}, New cumulative score: {new_score:.2f}")

        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Drop-off threshold reached on left extension. Stopping left extension.")
            break

        # Update positions and scores
        q_start = q_pos
        d_start = d_pos
        current_score = new_score
        if current_score > max_score:
            max_score = current_score

        #left_extension += 1

    # Extend to the right
    # right_extension = 1
    while q_end + 1 < len(q) and d_end + 1 < len(D):
        q_pos = q_end + 1
        d_pos = d_end + 1
        q_nuc = q[q_pos]
        d_probs = get_probs(d_pos)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s

        if verbose:
            print(f"Extending right to Q[{q_pos}]={q_nuc}, D[{d_pos}]")
            print(f"Expected score: {s:.2f}, New cumulative score: {new_score:.2f}")

        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Drop-off threshold reached on right extension. Stopping right extension.")
            break

        # Update positions and scores
        q_end = q_pos
        d_end = d_pos
        current_score = new_score
        if current_score > max_score:
            max_score = current_score

        #right_extension += 1

    if verbose:
        print(f"Final HSP: Q:{q_start}-{q_end}, D:{d_start}-{d_end}, Score: {max_score:.2f}")

    return (q_start, q_end, d_start, d_end, max_score)

def local_gapped_extension(q, D, conf_values, hsp, M, gap_penalty, verbose=False):
    """
    Perform a gapped extension (Smith-Waterman) on the given HSP.
    """
    q_start, q_end, d_start, d_end, hsp_score = hsp

    # Extract the relevant subsequences
    extended_q = q[q_start:q_end+1]
    extended_D = D[d_start:d_end+1]

    # Perform local gapped extension within this region
    len_q = len(extended_q)
    len_D = len(extended_D)

    # DP matrix and traceback
    dp = [[0]*(len_D+1) for _ in range(len_q+1)]
    traceback = [[None]*(len_D+1) for _ in range(len_q+1)]

    max_score = 0.0
    max_pos = (0,0)

    def get_probs(j):
        max_nuc = D[d_start + j -1]
        p_max = conf_values[d_start + j -1]
        return nucleotide_probabilities(p_max, max_nuc)

    # Fill DP using local alignment logic
    for i in range(1, len_q+1):
        q_nuc = extended_q[i-1]
        for j in range(1, len_D+1):
            dist = get_probs(j)
            match = dp[i-1][j-1] + expected_score(M, dist, q_nuc)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            cell_score = max(match, delete, insert, 0)

            dp[i][j] = cell_score
            if cell_score == 0:
                traceback[i][j] = None
            elif cell_score == match:
                traceback[i][j] = 'D'
            elif cell_score == delete:
                traceback[i][j] = 'U'
            else:
                traceback[i][j] = 'L'

            if cell_score > max_score:
                max_score = cell_score
                max_pos = (i, j)

    # Traceback from max_pos
    i, j = max_pos
    aligned_q = []
    aligned_D = []

    while i > 0 and j > 0 and dp[i][j] != 0:
        dir_ = traceback[i][j]
        if dir_ == 'D':
            aligned_q.append(extended_q[i-1])
            aligned_D.append(extended_D[j-1])
            i -= 1
            j -= 1
        elif dir_ == 'U':
            aligned_q.append(extended_q[i-1])
            aligned_D.append('-')
            i -= 1
        elif dir_ == 'L':
            aligned_q.append('-')
            aligned_D.append(extended_D[j-1])
            j -= 1
        else:
            break

    aligned_q.reverse()
    aligned_D.reverse()

    if verbose:
        print(f"\nGapped Extension on HSP Q:{q_start}-{q_end}, D:{d_start}-{d_end}")
        print(f"Extended Query: {''.join(aligned_q)}")
        print(f"Extended DB:    {''.join(aligned_D)}")
        print(f"Gapped Extension Score: {max_score:.2f}")

    return (max_score, ''.join(aligned_q), ''.join(aligned_D))

def main():
    sequence_file = './resources/fa2.txt'
    confidence_file = './resources/conf2.txt'
    substitution_matrix_file = './resources/substitution_matrix.txt'

    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)

    q = input("Enter your query sequence: ").strip().upper()
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))
    probability_threshold = float(input("Enter a probability threshold for w-mers (e.g. 0.0 for none): "))
    dropoff_threshold = float(input("Enter drop-off threshold for ungapped extension (e.g. 2.0): "))
    gap_penalty = float(input("Enter gap penalty (e.g. -2.0): "))
    verbose_input = input("Enable verbose mode? (y/n): ").strip().lower()
    verbose = True if verbose_input == 'y' else False

    # Compute background and query distributions
    background_probs = compute_background_distribution(D, conf_values)
    query_probs = compute_query_distribution(q)

    exp_score = compute_expected_score_overall(M, background_probs, query_probs)
    print(f"\nComputed expected score using query distribution and background distribution: {exp_score:.4f}")
    if exp_score < 0:
        print("The expected score is negative, which is often desirable.")
    else:
        print("The expected score is not negative. Consider adjusting the scoring matrix or approach.")

    # Build w-mer map
    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)

    # Find seeds
    seeds = find_seeds(q, f_map, w)

    print("\nFound seeds:")
    for seed in seeds:
        print(f"q_wmer: {seed[0]} q_pos: {seed[1]} d_pos: {seed[2]}")

    # Ungapped Extension Phase: Extend seeds into HSPs
    hsps = []
    for q_wmer, q_pos, d_pos in seeds:
        hsp = ungapped_extension(q, D, conf_values, q_pos, d_pos, w, M, dropoff_threshold, verbose=verbose)
        hsps.append(hsp)

    print("\nUngapped HSPs found:")
    for idx, hsp in enumerate(hsps):
        q_start, q_end, d_start, d_end, score = hsp
        print(f"HSP {idx+1}: Q:{q_start}-{q_end}, D:{d_start}-{d_end}, Score: {score:.2f}")

    # Select top HSPs (e.g., top 1)
    if not hsps:
        print("\nNo HSPs found. Exiting.")
        sys.exit(0)

    # Sort HSPs by score descending
    hsps_sorted = sorted(hsps, key=lambda x: x[4], reverse=True)
    top_hsps = hsps_sorted[:max_candidates]  # Adjust as needed

    # Gapped Extension Phase: Perform gapped extension on top HSPs
    gapped_alignments = []
    for hsp in top_hsps:
        score, align_q, align_d = local_gapped_extension(
            q, D, conf_values, hsp, M, gap_penalty, verbose=verbose
        )
        gapped_alignments.append((score, align_q, align_d))

    print("\nGapped Alignments found:")
    for idx, alignment in enumerate(gapped_alignments):
        score, align_q, align_d = alignment
        print(f"Gapped Alignment {idx+1}:")
        print(f"Alignment Score: {score:.2f}")
        print(f"Query: {align_q}")
        print(f"DB:    {align_d}\n")


def testing_map():
    sequence_file = './resources/fa2.txt'
    confidence_file = './resources/conf2.txt'
    substitution_matrix_file = './resources/substitution_matrix.txt'
    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))
    probability_threshold = float(input("Enter a probability threshold for w-mers (e.g. 0.0 for none): "))
    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)
    print(f_map)




if __name__ == "__main__":
    testing_map()

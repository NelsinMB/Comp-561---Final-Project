import sys
from collections import defaultdict, Counter

def load_sequence_and_confidence(seq_file, conf_file):
    """Load a sequence and corresponding confidence values."""
    with open(seq_file, 'r') as f:
        D = f.read().strip()

    with open(conf_file, 'r') as f:
        confidence_values = list(map(float, f.read().strip().split()))

    if len(D) != len(confidence_values):
        raise ValueError("Sequence and confidence lengths do not match.")

    return D, confidence_values

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

    rounds = min(3, len(p_max_list))  # up to 3 positions to vary
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

def find_seeds(q, f_map, w):
    seeds = []
    for start in range(len(q) - w + 1):
        q_wmer = q[start:start+w]
        if q_wmer in f_map:
            for pos in f_map[q_wmer]:
                seeds.append((q_wmer, start, pos))
    return seeds

def expected_score(M, probs, q_nuc):
    score = 0.0
    for x in ['A', 'C', 'G', 'T']:
        score += probs[x] * M[q_nuc][x]
    return score

def ungapped_extension(q, D, conf_values, seed_q_start, seed_d_start, w, M, dropoff_threshold, verbose=False):
    q_start = seed_q_start
    q_end = seed_q_start + w - 1
    d_start = seed_d_start
    d_end = seed_d_start + w - 1

    def get_probs(pos):
        max_nuc = D[pos]
        p_max = conf_values[pos]
        return nucleotide_probabilities(p_max, max_nuc)

    current_score = 0.0
    for offset in range(w):
        q_nuc = q[q_start + offset]
        d_probs = get_probs(d_start + offset)
        s = expected_score(M, d_probs, q_nuc)
        current_score += s
    max_score = current_score

    if verbose:
        print("\nStarting ungapped extension from seed:")
        print(f"  Seed q-range: {q_start}-{q_end}, D-range: {d_start}-{d_end}")
        print(f"  Initial seed score: {current_score:.2f} (max: {max_score:.2f})")

    # Extend left
    left_extension = 1
    while q_start - left_extension >= 0 and d_start - left_extension >= 0:
        q_pos_ext = q_start - left_extension
        d_pos_ext = d_start - left_extension
        q_nuc = q[q_pos_ext]
        d_probs = get_probs(d_pos_ext)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s
        if verbose:
            print(f"Extending left to q[{q_pos_ext}]={q_nuc}, D[{d_pos_ext}]")
            print(f"Probabilities: {d_probs}, Score Contribution: {s:.2f}")
            print(f"New Score: {new_score:.2f}, Max Score: {max_score:.2f}")
        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Dropoff threshold reached, stopping left extension.")
            break
        current_score = new_score
        if current_score > max_score:
            max_score = current_score
        q_start -= 1
        d_start -= 1
        left_extension += 1

    # Extend right
    right_extension = 1
    while q_end + right_extension < len(q) and d_end + right_extension < len(D):
        q_pos_ext = q_end + right_extension
        d_pos_ext = d_end + right_extension
        q_nuc = q[q_pos_ext]
        d_probs = get_probs(d_pos_ext)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s
        if verbose:
            print(f"Extending right to q[{q_pos_ext}]={q_nuc}, D[{d_pos_ext}]")
            print(f"Probabilities: {d_probs}, Score Contribution: {s:.2f}")
            print(f"New Score: {new_score:.2f}, Max Score: {max_score:.2f}")
        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Dropoff threshold reached, stopping right extension.")
            break
        current_score = new_score
        if current_score > max_score:
            max_score = current_score
        q_end += 1
        d_end += 1
        right_extension += 1

    if verbose:
        print(f"Final HSP: Q: {q_start}-{q_end}, D: {d_start}-{d_end}, Score: {max_score:.2f}\n")

    return (q_start, q_end, d_start, d_end, max_score)

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

def compute_expected_score(M, background_probs, query_probs):
    """
    E[score] = sum_{q_n,x} q_p(q_n)*bar{p}(x)*M(q_n,x)
    """
    expected = 0.0
    for q_n in ['A','C','G','T']:
        for x in ['A','C','G','T']:
            expected += query_probs[q_n]*background_probs[x]*M[q_n][x]
    return expected

if __name__ == "__main__":
    sequence_file = '/Users/nmarti55/Documents/fa2.txt'
    confidence_file = '/Users/nmarti55/Documents/conf2.txt'
    substitution_matrix_file = '/Users/nmarti55/Documents/substitution_matrix.txt'

    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)

    q = input("Enter your query sequence: ").strip()
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))
    probability_threshold = float(input("Enter a probability threshold for w-mers (e.g. 0.0 for none): "))
    dropoff_threshold = float(input("Enter drop-off threshold for ungapped extension (e.g. 2.0): "))
    verbose_input = input("Enable verbose mode? (y/n): ").strip().lower()
    verbose = True if verbose_input == 'y' else False

    # Compute background and query distributions
    background_probs = compute_background_distribution(D, conf_values)
    query_probs = compute_query_distribution(q)

    # Compute expected score
    exp_score = compute_expected_score(M, background_probs, query_probs)
    print(f"Computed expected score using query distribution and background distribution: {exp_score:.4f}")
    if exp_score < 0:
        print("The expected score is negative, which is often desirable.")
    else:
        print("The expected score is not negative. Consider adjusting the scoring matrix or approach.")

    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)
    seeds = find_seeds(q, f_map, w)

    print("Found seeds:")
    for seed in seeds:
        print("q_wmer:", seed[0], "q_pos:", seed[1], "d_pos:", seed[2])

    hsps = []
    for q_wmer, q_pos, d_pos in seeds:
        hsp = ungapped_extension(q, D, conf_values, q_pos, d_pos, w, M, dropoff_threshold, verbose=verbose)
        hsps.append(hsp)

    print("HSPs found:")
    for h in hsps:
        q_start, q_end, d_start, d_end, best_score = h
        print(f"Q: {q_start}-{q_end}, D: {d_start}-{d_end}, Score: {best_score:.2f}")

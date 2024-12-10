import sys
from collections import defaultdict
from itertools import product

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

    # Parse header
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
    equal_share = remaining_prob / 3.0
    probs = {max_nucleotide: p_max}
    for o in others:
        probs[o] = equal_share
    return probs

def get_w_mer(D, i, w):
    """Extract the w-mer starting at position i of D (0-based)."""
    if i + w > len(D):
        return None
    return D[i:i+w]

def generate_top_wmers_for_position(D, conf_values, i, w, max_candidates):
    """
    Generate up to max_candidates w-mers at position i.
    Heuristic:
    - Start with the maximum-likelihood w-mer.
    - Identify up to the three least probable positions in the w-mer and substitute 
      each with the other three nucleotides, generating variants.
    - Potentially produce up to 1 + 3*3 = 10 variants as before, but this time
      we limit ourselves to max_candidates.

    If max_candidates < 10, we simply take fewer variants.
    If max_candidates > 10, we still produce only up to 10 variants (or adjust 
    the logic to produce more variants if desired).
    """
    if i + w > len(D):
        return []

    # For each position in the w-mer, determine the probability distribution
    wmer_seq = D[i:i+w]
    wmer_probs = []
    p_max_list = []
    nucleotides = ['A', 'C', 'G', 'T']

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

    # If we want more candidates, generate variants.
    # We attempt up to 3 rounds, each substituting the nucleotides at the least probable positions.
    # Each round can generate up to 3 new variants.
    # Total up to 10 possible variants (1 original + 9 variants).
    # If max_candidates < 10, we'll just truncate sooner.
    # If max_candidates > 10, we could consider extending the logic, but for now we keep it simple.

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
        # Only generate variants from the original best_wmer for consistency
        if best_wmer in candidates:
            new_variants.extend(substitute_one_position(best_wmer, rel_pos))
        candidates.extend(new_variants)
        if len(candidates) >= max_candidates:
            break

    # Truncate to max_candidates if needed
    candidates = candidates[:max_candidates]

    # (Optional) Sort candidates by computed probability if desired. For now, we rely on heuristic.
    return candidates

def build_wmer_map(D, conf_values, w, max_candidates):
    """
    Build the hash map f: (w-mer) -> list of positions
    Insert up to max_candidates w-mers per position as per the chosen heuristic.
    """
    from collections import defaultdict
    f_map = defaultdict(list)
    for i in range(len(D) - w + 1):
        wmers = generate_top_wmers_for_position(D, conf_values, i, w, max_candidates)
        # Insert each w-mer into the map
        for wm in wmers:
            f_map[wm].append(i)
    return f_map

def find_seeds(q, f_map, w):
    """Find seed positions in D for each w-mer in q."""
    seeds = []
    for start in range(len(q) - w + 1):
        q_wmer = q[start:start+w]
        if q_wmer in f_map:
            for pos in f_map[q_wmer]:
                seeds.append((q_wmer, pos))
    return seeds

# Example usage:
if __name__ == "__main__":
    # Example inputs (modify paths as needed):
    sequence_file = '/Users/nmarti55/Documents/fa2.txt'
    confidence_file = '/Users/nmarti55/Documents/conf2.txt'
    substitution_matrix_file = '/Users/nmarti55/Documents/substitution_matrix.txt'

    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)

    q = input("Enter your query sequence: ").strip()
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))

    f_map = build_wmer_map(D, conf_values, w, max_candidates)
    seeds = find_seeds(q, f_map, w)

    print("Found seeds:")
    for s in seeds:
        print("q_wmer:", s[0], "position_in_D:", s[1])

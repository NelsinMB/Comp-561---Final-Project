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
        First line: A C G T (or chosen nucleotide order)
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

def most_probable_nucleotide(conf_value):
    """
    Given a confidence value which represents the probability of the most likely nucleotide,
    we must know which is the most likely nucleotide. Here, we assume that the sequence file
    gives us the predicted nucleotide at each position. The 'confidence' is the probability
    of that given nucleotide.

    If sequence[i] = N, and confidence[i] = p_max for that N,
    then other nucleotides have (1 - p_max)/3 each.
    """
    # This function just returns the conf_value, since we already know max nucleotide from D.
    return conf_value

def get_w_mer(D, i, w):
    """Extract the w-mer starting at position i of D (0-based)."""
    if i + w > len(D):
        return None
    return D[i:i+w]

def compute_wmer_probability(wmer, probs_list):
    """
    Given a w-mer and a list of probability distributions for each position:
    probs_list: a list of dicts [ {A: pA, C: pC, G: pG, T: pT}, ... ] of length w.
    Compute the probability as the product of probabilities for chosen nucleotides.
    """
    p = 1.0
    for idx, nuc in enumerate(wmer):
        p *= probs_list[idx][nuc]
    return p

def generate_top_wmers_for_position(D, conf_values, i, w):
    """
    Generate up to 10 w-mers at position i:
    - Start with the maximum-likelihood w-mer (choose for each position the given nucleotide in D).
    - Identify the positions in the w-mer with the smallest p_max[i].
    - Substitute those least probable positions with the other three nucleotides.

    Note: We assume that D[i] gives the max nucleotide at position i.
    """
    if i + w > len(D):
        return []

    # For each of the w positions, determine the probability distribution.
    wmer_seq = D[i:i+w]
    wmer_probs = []
    p_max_list = []
    for pos in range(i, i+w):
        # max nucleotide: D[pos], p_max = conf_values[pos]
        max_nuc = D[pos]
        p_max = conf_values[pos]
        dist = nucleotide_probabilities(p_max, max_nuc)
        wmer_probs.append(dist)
        p_max_list.append((pos - i, dist[max_nuc]))  # (relative_pos_in_wmer, p_max_value)

    # Sort w-mer positions by ascending p_max_value
    p_max_list.sort(key=lambda x: x[1]) 

    # Start with the maximum-likelihood w-mer:
    best_wmer = "".join(D[i+j] for j in range(w))
    candidates = [best_wmer]

    # We want up to 9 more variants by substituting:
    # Replace at the least probable position -> generate 3 variants
    # Replace at the second least probable position -> 3 variants
    # Replace at the third least probable position -> 3 variants
    # This yields up to 10 total (1 original + 9 variants).

    # Nucleotides set
    nucleotides = ['A', 'C', 'G', 'T']

    # A helper to create variants by substituting a single position:
    def substitute_one_position(wmer_str, rel_pos):
        orig_nuc = wmer_str[rel_pos]
        # Pick other three nucleotides:
        variants = []
        for nuc in nucleotides:
            if nuc != orig_nuc:
                var = wmer_str[:rel_pos] + nuc + wmer_str[rel_pos+1:]
                variants.append(var)
        return variants

    # Create variants
    for idx_in_sorted in range(min(3, len(p_max_list))):
        rel_pos = p_max_list[idx_in_sorted][0]
        new_variants = []
        for c in candidates:
            # Only generate variants from the original best_wmer for simplicity,
            # as discussed in the approach. If you want a more exhaustive approach,
            # you could consider generating variants from previously generated variants.
            if c == best_wmer:
                new_variants.extend(substitute_one_position(c, rel_pos))
        # Add these variants to candidates, ensuring we don't exceed the limit.
        candidates.extend(new_variants)
        if len(candidates) >= 10:
            break

    # If we somehow got more than 10, truncate:
    candidates = candidates[:10]

    # Optional: We can rank candidates by their computed probability and pick the top 10.
    # Since we rely on the heuristic, we assume all candidates are decent.
    # If needed, compute probabilities and sort:
    # candidate_probs = []
    # for c in candidates:
    #     p = compute_wmer_probability(c, wmer_probs)
    #     candidate_probs.append((c, p))
    # candidate_probs.sort(key=lambda x: x[1], reverse=True)
    # candidates = [cp[0] for cp in candidate_probs][:10]

    return candidates


def build_wmer_map(D, conf_values, w):
    """
    Build the hash map f: (w-mer) -> list of positions
    Insert up to 10 w-mers per position as per the chosen heuristic.
    """
    f_map = defaultdict(list)
    for i in range(len(D) - w + 1):
        wmers = generate_top_wmers_for_position(D, conf_values, i, w)
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

    # Load D and confidences
    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)

    # Load substitution matrix (not heavily used here yet)
    M = load_substitution_matrix(substitution_matrix_file)

    # Ask for query q and w size (for demonstration):
    q = input("Enter your query sequence: ").strip()
    w = int(input("Enter word size w: "))

    # Build w-mer map
    f_map = build_wmer_map(D, conf_values, w)

    # Find seeds
    seeds = find_seeds(q, f_map, w)

    print("Found seeds:")
    for s in seeds:
        print("q_wmer:", s[0], "position_in_D:", s[1])

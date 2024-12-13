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
    Tracks and returns the alignment corresponding to the maximum score.
    """
    # Initialize seed boundaries
    q_start = seed_q_start
    q_end = seed_q_start + w - 1
    d_start = seed_d_start
    d_end = seed_d_start + w - 1

    def get_probs(pos):
        max_nuc = D[pos]
        p_max = conf_values[pos]
        return nucleotide_probabilities(p_max, max_nuc)

    # Score the seed using the expected score of each position
    current_score = 0.0
    for offset in range(w):
        q_nuc = q[q_start + offset]
        d_pos = d_start + offset
        d_nuc = D[d_pos]
        d_probs = get_probs(d_pos)
        s = expected_score(M, d_probs, q_nuc)
        if verbose:
            print(f"Position Q[{q_start + offset}]={q_nuc} - D[{d_pos}]={d_nuc}, Expected Score: {s:.2f}")
        current_score += s

    max_score = current_score
    # Initialize best alignment positions
    best_q_start, best_q_end = q_start, q_end
    best_d_start, best_d_end = d_start, d_end

    if verbose:
        print(f"\nStarting ungapped extension for seed at Q:{q_start}-{q_end}, D:{d_start}-{d_end}")
        print(f"Initial ungapped score: {current_score:.2f}")

    # Extend to the left
    while q_start > 0 and d_start > 0:
        q_pos = q_start - 1
        d_pos = d_start - 1
        q_nuc = q[q_pos]
        d_nuc = D[d_pos]
        d_probs = get_probs(d_pos)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s

        if verbose:
            print(f"Extending left to Q[{q_pos}]={q_nuc}, D[{d_pos}]={d_nuc}")
            print(f"Expected score: {s:.2f}, New cumulative score: {new_score:.2f}")

        if new_score > max_score:
            max_score = new_score
            best_q_start, best_d_start = q_pos, d_pos
            best_q_end, best_d_end = q_end, d_end

        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Drop-off threshold reached on left extension. Stopping left extension.")
            break

        # Update positions and scores
        q_start = q_pos
        d_start = d_pos
        current_score = new_score

    # Extend to the right
    while q_end + 1 < len(q) and d_end + 1 < len(D):
        q_pos = q_end + 1
        d_pos = d_end + 1
        q_nuc = q[q_pos]
        d_nuc = D[d_pos]
        d_probs = get_probs(d_pos)
        s = expected_score(M, d_probs, q_nuc)
        new_score = current_score + s

        if verbose:
            print(f"Extending right to Q[{q_pos}]={q_nuc}, D[{d_pos}]={d_nuc}")
            print(f"Expected score: {s:.2f}, New cumulative score: {new_score:.2f}")

        if new_score > max_score:
            max_score = new_score
            best_q_end, best_d_end = q_pos, d_pos

        if new_score < max_score - dropoff_threshold:
            if verbose:
                print("Drop-off threshold reached on right extension. Stopping right extension.")
            break

        # Update positions and scores
        q_end = q_pos
        d_end = d_pos
        current_score = new_score

    if verbose:
        aligned_query = q[best_q_start:best_q_end+1]
        aligned_db = D[best_d_start:best_d_end+1]
        print(f"Final HSP: Q:{best_q_start}-{best_q_end} = '{aligned_query}' | "
              f"D:{best_d_start}-{best_d_end} = '{aligned_db}' | Score: {max_score:.2f}")

    return (best_q_start, best_q_end, best_d_start, best_d_end, max_score)


def local_gapped_extension(q, D, conf_values, hsp, M, gap_penalty, dropoff_threshold, verbose=False):
    """
    Perform a gapped extension (Smith-Waterman) dynamically extending beyond HSP bounds,
    with a stopping condition based on the score falling more than the dropoff_threshold below the max score so far.
    
    Parameters:
        q (str): Query sequence.
        D (str): Database sequence.
        conf_values (List[float]): Confidence values for each nucleotide in D.
        hsp (Tuple[int, int, int, int, float]): HSP represented as (q_start, q_end, d_start, d_end, score).
        M (Dict[str, Dict[str, float]]): Substitution matrix.
        gap_penalty (float): Penalty for introducing a gap.
        dropoff_threshold (float): Threshold to stop extension when score drops below (max_score_so_far - threshold).
        verbose (bool): If True, prints detailed debug information.
        
    Returns:
        Tuple[float, str, str]: (max_score, aligned_query, aligned_database)
    """

    q_start, q_end, d_start, d_end, hsp_score = hsp

    # Initialize variables
    len_q = len(q)
    len_D = len(D)
    
    # Initialize DP matrix
    dp = [[0] * (len_D + 1) for _ in range(len_q + 1)]
    
    max_score_so_far = 0.0
    max_pos = None
    early_stop = False

    def get_probs(j):
        """
        Retrieve nucleotide probabilities for position j in D.
        """
        max_nuc = D[j - 1]
        p_max = conf_values[j - 1]
        return nucleotide_probabilities(p_max, max_nuc)

    # Fill DP matrix with dynamic programming
    for i in range(1, len_q + 1):
        q_nuc = q[i - 1]
        current_row_max = 0.0  # Reset for the current row
        
        for j in range(1, len_D + 1):
            dist = get_probs(j)
            match = dp[i - 1][j - 1] + expected_score(M, dist, q_nuc)
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(0, match, delete, insert)

            # Update current row's max score
            if dp[i][j] > current_row_max:
                current_row_max = dp[i][j]
            
            # Update global max score and position
            if dp[i][j] > max_score_so_far:
                max_score_so_far = dp[i][j]
                max_pos = (i, j)

        # Check the drop-off condition after processing the row
        if current_row_max < (max_score_so_far - dropoff_threshold):
            if verbose:
                print(f"Stopping extension: Current row max {current_row_max:.2f} is below (max_score_so_far - threshold) "
                      f"{max_score_so_far - dropoff_threshold:.2f}")
            early_stop = True
            break

    # If early stopping was triggered, return no alignment
    if early_stop or max_pos is None:
        if verbose:
            print("Early termination: No significant alignment found within the drop-off threshold.")
        return 0.0, '', ''

    # Traceback to retrieve the alignment
    aligned_q = []
    aligned_D = []
    i, j = max_pos
    while i > 0 and j > 0 and dp[i][j] > 0:
        current_score = dp[i][j]
        score_current = expected_score(M, get_probs(j), q[i - 1])
        
        if current_score == dp[i - 1][j - 1] + score_current:
            aligned_q.append(q[i - 1])
            aligned_D.append(D[j - 1])
            i -= 1
            j -= 1
        elif current_score == dp[i - 1][j] + gap_penalty:
            aligned_q.append(q[i - 1])
            aligned_D.append('-')
            i -= 1
        elif current_score == dp[i][j - 1] + gap_penalty:
            aligned_q.append('-')
            aligned_D.append(D[j - 1])
            j -= 1
        else:
            break  # This condition should not occur if DP is correctly filled

    # Reverse to get the correct alignment
    aligned_q.reverse()
    aligned_D.reverse()

    # Optional: Calculate the starting positions (can be used for further analysis)
    query_start = i
    db_start = j

    if verbose:
        print(f"\nGapped Extension on HSP Q:{q_start}-{q_end}, D:{d_start}-{d_end}")
        print(f"Extended Query: {''.join(aligned_q)}")
        print(f"Extended DB:    {''.join(aligned_D)}")
        print(f"Gapped Extension Score: {max_score_so_far:.2f}")

    return max_score_so_far, ''.join(aligned_q), ''.join(aligned_D)

def remove_duplicate_hsps(hsps):
    """
    Remove duplicate HSPs from the list.

    Parameters:
        hsps (List[Tuple[int, int, int, int, float]]): List of HSPs.

    Returns:
        List[Tuple[int, int, int, int, float]]: List of unique HSPs.
    """
    seen = set()
    unique_hsps = []
    for hsp in hsps:
        # Create a hashable key for the HSP
        key = (hsp[0], hsp[1], hsp[2], hsp[3], round(hsp[4], 4))  # Rounded score to handle floating point precision
        if key not in seen:
            seen.add(key)
            unique_hsps.append(hsp)
    return unique_hsps


def filter_identical_alignments(alignments):
    """
    Filter identical gapped alignments to retain only unique ones.
    """
    unique_alignments = []
    seen = set()

    for alignment in alignments:
        score, align_q, align_d = alignment
        key = (align_q, align_d)  # Use alignment sequences as the key
        if key not in seen:
            unique_alignments.append(alignment)
            seen.add(key)

    return unique_alignments



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
    
    return D, conf_values, M, w, max_candidates, probability_threshold, f_map

def testing_seeding():
    q = input("Enter your query sequence: ").strip().upper()
    D, conf_values, M, w, max_candidates, probability_threshold, f_map = testing_map()
    verbose_input = input("Enable verbose mode? (y/n): ").strip().lower()
    verbose = True if verbose_input == 'y' else False

    # Compute background and query distributions
    background_probs = compute_background_distribution(D, conf_values)
    query_probs = compute_query_distribution(q)

    # Find seeds
    seeds = find_seeds(q, f_map, w)

    print("\nFound seeds:")
    for seed in seeds:
        print(f"q_wmer: {seed[0]} q_pos: {seed[1]} d_pos: {seed[2]}")

    return q, D, conf_values, M, w, max_candidates, probability_threshold, f_map, verbose, seeds


def testing_ungapped_extension():
    """
    Tests the ungapped extension process with verbose output using sample data.
    """
    q, D, conf_values, M, w, max_candidates, probability_threshold, f_map, verbose, seeds = testing_seeding()
    
    # Define drop-off threshold
    dropoff_threshold = float(input("Enter drop-off threshold for ungapped extension (e.g. 2.0): "))
    
    print("\n--- Ungapped Extension Phase ---")
    
    # Ungapped Extension Phase: Extend seeds into HSPs
    hsps = []
    for idx, (q_wmer, q_pos, d_pos) in enumerate(seeds):

        if verbose: 
            print(f"\nProcessing Seed {idx+1}:")
            print(f"  Query W-mer: '{q_wmer}' at Q[{q_pos}:{q_pos + w -1}]")
            print(f"  Database Position: D[{d_pos}:{d_pos + w -1}]")
        
        hsp = ungapped_extension(
            q, D, conf_values, q_pos, d_pos, w, M, dropoff_threshold, verbose=verbose
        )
        hsps.append(hsp)

        hsps = remove_duplicate_hsps(hsps)

    
    print("\n--- Ungapped HSPs Found ---")
    for idx, hsp in enumerate(hsps):
        q_start, q_end, d_start, d_end, score = hsp
        print(f"HSP {idx+1}: Q[{q_start}:{q_end}] = '{q[q_start:q_end+1]}' | "
              f"D[{d_start}:{d_end}] = '{D[d_start:d_end+1]}' | Score: {score:.2f}")
    
    # Select top HSPs (e.g., top 1)
    if not hsps:
        print("\nNo HSPs found. Exiting.")
        sys.exit(0)
    
    # Sort HSPs by score descending
    hsps = sorted(hsps, key=lambda x: x[4], reverse=True)[:5]
    
        
    return q, D, conf_values, hsps, M, verbose

def testing_gapped_extension():
    q, D, conf_values, hsps, M, verbose = testing_ungapped_extension()

    gap_penalty = float(input("Enter gap penalty (e.g. -2.0): "))
    max_hsps = int(input("Enter the maximum number of HSPs to perform gapped extension on (-1 for all): "))

    # Not sure if this is needed.
    if max_hsps == -1:
        max_hsps = len(hsps)
    hsps = hsps[:max_hsps]

    # Gapped Extension Phase: Perform gapped extension on top HSPs
    gapped_alignments = []
    for hsp in hsps:
        score, align_q, align_d = local_gapped_extension(
            q, D, conf_values, hsp, M, gap_penalty, dropoff_threshold=10.0, verbose=verbose
        )
        gapped_alignments.append((score, align_q, align_d))

    filtered_gapped_alignments = filter_identical_alignments(gapped_alignments)

    print("\nGapped Alignments found:")
    for idx, alignment in enumerate(filtered_gapped_alignments):
        score, align_q, align_d = alignment
        print(f"Gapped Alignment {idx+1}:")
        print(f"Alignment Score: {score:.2f}")
        print(f"Query: {align_q}")
        print(f"DB:    {align_d}\n")

def main():
    sequence_file = './resources/fa2.txt'
    confidence_file = './resources/conf2.txt'
    substitution_matrix_file = './resources/substitution_matrix.txt'
    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))
    probability_threshold = float(input("Enter a probability threshold for w-mers (e.g. 0.0 for none): "))
    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)

    q = input("Enter your query sequence: ").strip().upper()
    verbose_input = input("Enable verbose mode? (y/n): ").strip().lower()
    verbose = True if verbose_input == 'y' else False

    # Compute background and query distributions
    background_probs = compute_background_distribution(D, conf_values)
    query_probs = compute_query_distribution(q)

    # Find seeds
    seeds = find_seeds(q, f_map, w)

    print("\nFound seeds:")
    for seed in seeds:
        print(f"q_wmer: {seed[0]} q_pos: {seed[1]} d_pos: {seed[2]}")

    dropoff_threshold = float(input("Enter drop-off threshold for ungapped extension (e.g. 2.0): "))
    
    print("\n--- Ungapped Extension Phase ---")
    
    # Ungapped Extension Phase: Extend seeds into HSPs
    hsps = []
    for idx, (q_wmer, q_pos, d_pos) in enumerate(seeds):

        if verbose: 
            print(f"\nProcessing Seed {idx+1}:")
            print(f"  Query W-mer: '{q_wmer}' at Q[{q_pos}:{q_pos + w -1}]")
            print(f"  Database Position: D[{d_pos}:{d_pos + w -1}]")
        
        hsp = ungapped_extension(
            q, D, conf_values, q_pos, d_pos, w, M, dropoff_threshold, verbose=verbose
        )
        hsps.append(hsp)

        hsps = remove_duplicate_hsps(hsps)

    
    print("\n--- Ungapped HSPs Found ---")
    for idx, hsp in enumerate(hsps):
        q_start, q_end, d_start, d_end, score = hsp
        print(f"HSP {idx+1}: Q[{q_start}:{q_end}] = '{q[q_start:q_end+1]}' | "
              f"D[{d_start}:{d_end}] = '{D[d_start:d_end+1]}' | Score: {score:.2f}")
    
    # Select top HSPs (e.g., top 1)
    if not hsps:
        print("\nNo HSPs found. Exiting.")
        sys.exit(0)
    

    # Sort HSPs by score descending
    hsps = sorted(hsps, key=lambda x: x[4], reverse=True)

    gap_penalty = float(input("Enter gap penalty (e.g. -2.0): "))
    max_hsps = int(input("Enter the maximum number of HSPs to perform gapped extension on (-1 for all): "))

    # Not sure if this is needed.
    if max_hsps == -1:
        max_hsps = len(hsps)
    hsps = hsps[:max_hsps]

    # Gapped Extension Phase: Perform gapped extension on top HSPs
    gapped_alignments = []
    for hsp in hsps:
        score, align_q, align_d = local_gapped_extension(
            q, D, conf_values, hsp, M, gap_penalty, dropoff_threshold=10.0, verbose=verbose
        )
        gapped_alignments.append((score, align_q, align_d))

    filtered_gapped_alignments = filter_identical_alignments(gapped_alignments)

    print("\nGapped Alignments found:")
    for idx, alignment in enumerate(filtered_gapped_alignments):
        score, align_q, align_d = alignment
        print(f"Gapped Alignment {idx+1}:")
        print(f"Alignment Score: {score:.2f}")
        print(f"Query: {align_q}")
        print(f"DB:    {align_d}\n")

import random

def generate_query_from_probabilistic_genome(D, conf_values, query_length, num_mutations=2, num_indels=1):

    
    """
    Generate a query sequence from a probabilistic genome.

    Parameters:
        D (str): Probabilistic genome (most likely nucleotide sequence).
        conf_values (List[float]): Confidence values for each position in D.
        query_length (int): Length of the query sequence.
        num_mutations (int): Number of random substitutions to introduce.
        num_indels (int): Number of random indels to introduce.

    Returns:
        str: Generated query sequence with mutations and indels.
        int: Starting position of the query in the original genome.
    """
    nucleotides = ['A', 'C', 'G', 'T']

    # Step 1: Pick a random starting position for the query
    start_pos = random.randint(0, len(D) - query_length)
    query = []

    # Step 2: Generate the query sequence based on probabilities
    for i in range(start_pos, start_pos + query_length):
        max_nuc = D[i]
        p_max = conf_values[i]
        probs = nucleotide_probabilities(p_max, max_nuc)
        query.append(random.choices(nucleotides, weights=[probs[n] for n in nucleotides])[0])

    # Step 3: Introduce random substitutions
    query = list(query)
    mutation_indices = random.sample(range(query_length), min(num_mutations, query_length))
    for idx in mutation_indices:
        original_nuc = query[idx]
        query[idx] = random.choice([n for n in nucleotides if n != original_nuc])

    # Step 4: Introduce random indels
    for _ in range(num_indels):
        if random.random() < 0.5 and len(query) > 1:  # Perform deletion
            del_idx = random.randint(0, len(query) - 1)
            query.pop(del_idx)
        else:  # Perform insertion
            ins_idx = random.randint(0, len(query))
            query.insert(ins_idx, random.choice(nucleotides))

    return ''.join(query), start_pos

def test_algorithm(D, conf_values, query_length, num_mutations=2, num_indels=1, verbose=False):
    """
    Test the sequence alignment algorithm with a probabilistically generated query sequence.

    Parameters:
        D (str): Probabilistic genome (most likely nucleotide sequence).
        conf_values (List[float]): Confidence values for each position in D.
        query_length (int): Length of the query sequence.
        num_mutations (int): Number of random substitutions to introduce.
        num_indels (int): Number of random indels to introduce.
        verbose (bool): Whether to print detailed output during the test.

    Returns:
        None
    """
    # Generate query sequence
    query, true_start_pos = generate_query_from_probabilistic_genome(D, conf_values, query_length, num_mutations, num_indels)

    if verbose:
        print(f"Generated query sequence: {query}")
        print(f"True start position in genome: {true_start_pos}")

    # Load a substitution matrix (you may need to provide your own matrix file)
    substitution_matrix_file = './resources/substitution_matrix.txt'
    M = load_substitution_matrix(substitution_matrix_file)

    # Build the w-mer map for the probabilistic genome
    w = 11  # Word size (adjust as needed)
    max_candidates = 10  # Number of sequences to consider per position
    probability_threshold = 0.1  # Adjust as needed
    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)

    # Find seeds
    seeds = find_seeds(query, f_map, w)

    if verbose:
        print("\nSeeds found:")
        for seed in seeds:
            print(f"Query w-mer: {seed[0]} | Query pos: {seed[1]} | Genome pos: {seed[2]}")

    # Perform ungapped extension
    dropoff_threshold = 10.0
    hsps = []
    for q_wmer, q_pos, d_pos in seeds:
        hsp = ungapped_extension(query, D, conf_values, q_pos, d_pos, w, M, dropoff_threshold, verbose)
        hsps.append(hsp)

    # Remove duplicates and sort by score
    hsps = remove_duplicate_hsps(hsps)
    hsps = sorted(hsps, key=lambda x: x[4], reverse=True)

    if verbose:
        print("\nUngapped HSPs:")
        for hsp in hsps:
            print(hsp)

    # Perform gapped extension
    gap_penalty = -2.0
    gapped_alignments = []
    for hsp in hsps:
        alignment = local_gapped_extension(query, D, conf_values, hsp, M, gap_penalty, dropoff_threshold, verbose)
        gapped_alignments.append(alignment)

    # Filter unique alignments
    gapped_alignments = filter_identical_alignments(gapped_alignments)

    if verbose:
        print("\nGapped alignments:")
        for alignment in gapped_alignments:
            print(f"Score: {alignment[0]}\nQuery: {alignment[1]}\nGenome: {alignment[2]}\n")

    # Check if the true starting position is captured in the alignments
    alignment_found = any(true_start_pos in range(hsp[2], hsp[3] + 1) for hsp in hsps)
    if alignment_found:
        print("\nThe algorithm successfully identified the correct portion of the genome.")
    else:
        print("\nThe algorithm failed to identify the correct portion of the genome.")


if __name__ == "__main__":
    sequence_file = './resources/long.txt'
    confidence_file = './resources/long_conf.txt'
    substitution_matrix_file = './resources/substitution_matrix.txt'
    D, conf_values = load_sequence_and_confidence(sequence_file, confidence_file)
    M = load_substitution_matrix(substitution_matrix_file)
    w = int(input("Enter word size w: "))
    max_candidates = int(input("Enter how many sequences you want to consider at each index: "))
    probability_threshold = float(input("Enter a probability threshold for w-mers (e.g. 0.0 for none): "))
    f_map = build_wmer_map(D, conf_values, w, max_candidates, probability_threshold)



    query_length = 1000
    num_mutations = 20
    num_indels = 20
    test_algorithm(D,conf_values, query_length, num_mutations, num_indels, True)
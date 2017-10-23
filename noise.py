import sim_data as sd
import NJ
def join_noised(n_seq, k, com=.05):
    data = sd.mutate_samples_uniform(n_seq, k, com)
    return NJ.join(data)

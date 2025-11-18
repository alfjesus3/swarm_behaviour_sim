import numpy as np
import logging

# A logger for this file
log = logging.getLogger(__name__)

'''
class RunningMeanStd(object):
    """
            (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)
    Adapted from https://github.com/ALRhub/deep_rl_for_swarms/blob/master/deep_rl_for_swarms/common/running_mean_std.py 
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count    


    ###
    # ... To be moved ...
    ###

    tmp_eps = 1e-8

    # Mean position embedding by swarm state
    mean_embeds1, mean_embeds2 = [], []
    for i in range(idxs[-1][0]):
        mean_e1 = RunningMeanStd(epsilon=tmp_eps, shape=(boid_pos1.shape[-1]+boid_vels1.shape[-1],))
        mean_e2 = RunningMeanStd(epsilon=tmp_eps, shape=(boid_pos2.shape[-1]+1,))

        mean_e1.update(np.concatenate((boid_pos1[i], boid_vels1[i]), axis=1))
        mean_e2.update(np.concatenate((boid_pos2[i], boid_vels2[i].reshape(-1, 1)), axis=1))

        # x = np.concatenate([x1, x2, x3], axis=0)
        # ms1 = [x.mean(axis=0), x.var(axis=0)]
        # rms.update(x1)
        # rms.update(x2)
        # rms.update(x3)
        # ms2 = [rms.mean, rms.var]
        # assert np.allclose(ms1, ms2)

        mean_embeds1.append(mean_e1)
        mean_embeds2.append(mean_e2)

    # TODO plot only the (x,y) distribution embedding part and add to a .gif

'''


# TODO
#   Consider implementing Radial basis kernel function to explore structure (and symmetries) ... \url{https://medium.com/@zekiemretekin/practical-example-of-clustering-and-radial-basis-functions-rbf-629dc3ece275}
#   Consider fixing and cleaning (from above) theClassical and NN mean swarm embedding; (TODO understand if this work for non-euclidean spaces?!)
#       Here’s the code of the Deep mean embedding neural network to be training for the experiments https://github.com/ALRhub/deep_rl_for_swarms/blob/master/deep_rl_for_swarms/policies/mean_embedding.py
#   KL divergence in between expert and learned distributions
#       action-space similarities like KL divergence and cosine similarity

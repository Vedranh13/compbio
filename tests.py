import unittest
import JC
import utils
import NJ
import sim_data
import numpy as np
import noise


class TestJCMethods(unittest.TestCase):

    def test_evolve(self):
        base = utils.gen_base_strand(15)
        # tree = [10, [5, [15], [15, [3], [3]]], [5]]
        tree = [100, [50, [], [100, [], []]], [150, [], []]]
        data = JC.evolve(base, tree, .005)
        # print(base)
        self.assertEqual(len(data), 5)
        # uncomment to see simulated data
        # print(data)
        # for left in data:
        #     print(utils.jc_distance(base, left))

    def test_newick_cherry(self):
        true = "(1:0.5,3:0.2)"
        self.assertEqual(utils.newick_cherry(1, 3, .5, .2), true)

    def test_in_out(self):
        N = 5
        in_to_real = {a: a for a in range(1, N + 1)}
        for f, g, DFU, DGU in [(1, 2, .5, .6), (1, 3, .55, .63), (2, 3, .2, .8), (1, 2, .9, .91)]:
            spec1 = in_to_real[f]
            spec2 = in_to_real[g]
            for spec in range(g, N):
                in_to_real[spec] = in_to_real[spec + 1]
            in_to_real[f] = utils.newick_cherry(spec1, spec2, DFU, DGU)
        true = utils.newick_cherry(1, 2, .5, .6)
        true = utils.newick_cherry(true, 4, .55, .63)
        true = utils.newick_cherry(true, utils.newick_cherry(3, 5, .2, .8), .9, .91)
        self.assertEqual(in_to_real[1], true)

    def test_create_k_copies(self):
        np.random.seed(1512)
        cs61a = [100, [50, [], []], [100, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .005)
        nk = sim_data.create_k_copies(strands, 5)
        self.assertEqual(len(nk), 20)
        self.assertEqual(len(set(nk)), 4)

    def test_k_samples_uniform(self):
        """Takes in a simple tree and simulates having k noisy
        samples from each of the n true species and perform NJ as if
        it was nk species. We should see n subtrees of k members each. We
        see almost that, with a few species crossing over, which makes
        sense because we do have information loss and we have seen below
        how NJ can be sensative to uniform noise"""
        np.random.seed(1512)
        cs61a = [100, [50, [], []], [100, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .005)
        utils.draw(noise.join_noised(strands, 5), title="4 species with 5 noisy samples")

    def test_simulate_error_extinct(self):
        """Want a tree like     A
                            |           |

                        |            C     D
                    B
                            |
                        E       F
        Basically, this test does neighboor joining when some of the species dies long before others"""
        np.random.seed(9001)
        cs61a = [100, [50, [], [100, [], []]], [150, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .0005)
        utils.draw(NJ.join(strands), title="Extinct species tree")

    def test_simulate_simple(self):
        """Want a tree like     A
                            |           |

                        E       F      C     D
            Very simple, standard tree"""
        np.random.seed(9001)
        cs61a = [100, [50, [], []], [100, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .005)
        utils.draw(NJ.join(strands), title="Simple 4 leaf tree")

    def test_simulate_simple_big(self):
        """Want a tree like     A
                            |           |

                        E       F      C     D ...
            Very simple, standard tree, but with many leaves.
            Here is the graphical output from my computer:
            http://etetoolkit.org/treeview/?treeid=1608e9140c43b5ff34999961ef5ed425&algid="""
        np.random.seed(9001)
        cs61a = [100, [50, [15, [200, [150, [], []], [150, [], []]], [200, [150, [], []], [150, [], []]]], [15, [200, [150, [], []], [150, [], []]], [200, [150, [], []], [150, [], []]]]], [50, [15, [200, [150, [], []], [150, [], []]], [200, [150, [], []], [150, [], []]]], [15, [200, [150, [], []], [150, [], []]], [200, [150, [], []], [150, [], []]]]]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .0005)
        print("32 Leaf Tree:\n", NJ.join(strands))

    def test_simulate_simple_uni5(self):
        """Want a tree like     A
                            |           |

                        E       F      C     D
            Very simple, standard tree, but with five percent uniform corruption.
            As you should see, this level noise affects the distances, but not the topology.
            Note however that in certain cases it could affect both due to chance mutation patterns
            that are extra bad. In the process of testing this, the percent of times this occurs is
            slightly less than 50, showing that this implementation of neighboor joining is quite
            suspestible to error, even with just 5% corruption"""
        np.random.seed(1512)
        cs61a = [100, [100, [], []], [100, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .005)
        for i in range(len(strands)):
            strands[i] = sim_data.mutate_str_uniform(strands[i], .05)
        print("5 % corrupted tree:\n", NJ.join(strands))

    def test_simulate_simple_uni10(self):
        """Want a tree like     A
                            |           |

                        E       F      C     D
            Very simple, standard tree, but with five percent uniform corruption.
            As you should see, this level noise affects both distances and topologies, intermixing
            leaves from the left and right subtrees. Again, with only 10%, we see a major
            deviance from the true tree, showing the sensitivity of neighboor joining."""
        np.random.seed(1512)
        cs61a = [100, [100, [], []], [100, [], []]]
        strands = JC.evolve(utils.gen_base_strand(30), cs61a, .005)
        print("True Tree:\n", NJ.join(strands))
        for i in range(len(strands)):
            strands[i] = sim_data.mutate_str_uniform(strands[i], .1)
        print("10 % corrupted tree:\n", NJ.join(strands))


if __name__ == '__main__':
    unittest.main()

import unittest
import JC
import utils
import NJ

class TestJCMethods(unittest.TestCase):

    def test_evolve(self):
        base = utils.gen_base_strand(15)
        #tree = [10, [5, [15], [15, [3], [3]]], [5]]
        tree = [100, [50, [], [100, [], []]], [150, [], []]]
        data = JC.evolve(base, tree, .005)
        print(base)
        self.assertEqual(len(data), 5)
        print(data)
        for left in data:
            print(utils.jc_distance(base, left))


    def test_newick_cherry(self):
        true = "(1:0.5,3:0.2)"
        self.assertEqual(utils.newick_cherry(1, 3, .5, .2), true)


    def test_in_out(self):
        N = 5
        in_to_real = {a : a for a in range(1, N + 1)}
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


    def test_simulate(self):
        """Want a tree like     A
                            |           |

                        |            C     D
                    B
                            |
                        E       F"""
        cs61a = [100, [50, [], [100, [], []]], [150, [], []]]
        strands = JC.evolve(utils.gen_base_strand(15), cs61a, .005)
        print(NJ.join(strands))
if __name__ == '__main__':
    unittest.main()

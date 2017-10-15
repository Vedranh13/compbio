import unittest
import JC
import utils

class TestJCMethods(unittest.TestCase):

    def test_evolve(self):
        base = utils.gen_base_strand(15)
        tree = [10, [5, [15], [15, [3], [3]]], [5]]
        data = JC.evolve(base, tree, .05)
        print(base)
        self.assertEqual(len(data), 4)
        print(data)
        for left in data:
            print(utils.jc_distance(base, left))
if __name__ == '__main__':
    unittest.main()

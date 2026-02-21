import unittest
import math

class TestTopologicalFractionalization(unittest.TestCase):
    def test_witten_effect_fractions(self):
        """
        Tests if the Z_3 (three-fold) topological phase angles
        correctly generate exact 1/3 and 2/3 charge fractions via Witten Effect.
        """
        allowed_thetas = [0, 2*math.pi/3, -2*math.pi/3, 4*math.pi/3, -4*math.pi/3]
        generated_fractions = set()
        
        for t in allowed_thetas:
            for base_n in [0, 1, -1]:
                val = base_n + (t / (2 * math.pi))
                if abs(val) > 0.01 and abs(abs(val) - 1.0) > 0.01 and abs(val) < 1.0:
                    generated_fractions.add(round(val, 4))
                    
        expected = {round(1/3, 4), round(2/3, 4), round(-1/3, 4), round(-2/3, 4)}
        self.assertEqual(generated_fractions, expected, 
                         "Witten Effect failed to recover precise quark charge fractions.")

if __name__ == '__main__':
    unittest.main()
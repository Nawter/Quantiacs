import unittest

# unittest docs
# https://docs.python.org/2/library/unittest.html

# assert docs
# https://docs.python.org/2/library/unittest.html#assert-methods

# string.index docs
# https://docs.python.org/2/library/string.html#string.index

class TestIndex(unittest.TestCase):

    # Run before each test
    def setUp(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.repeating_string = "abababab"

    def test_my_first_test(self):
        self.assertEqual(self.alphabet.index('ab'), 0)
        self.assertEqual(self.repeating_string.index('ab'), 0)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            self.alphabet.index('not_in_the_alphabet')

    def test_index_substring_types_KO(self):
        with self.assertRaises(TypeError):
            self.alphabet.index(12)

    def test_find_substring_OK(self):
        self.assertEqual(self.alphabet.find('ab'), 0)
        self.assertEqual(self.repeating_string.find('ab'), 0)

    def test_find_substring_KO(self):
            self.assertNotEqual(self.alphabet.find('zn'), 0)
            self.assertNotEqual(self.repeating_string.index('zn'), 0)

    def test_find_substring_numeric_string_KO(self):
            self.assertNotEqual(self.alphabet.find('12'), 0)
            self.assertNotEqual(self.repeating_string.index('12'), 0)

    def test_find_substring_types_KO(self):
        with self.assertRaises(TypeError):
            self.alphabet.find(12)



    # Run after each test
    def tearDown(self):
        return

if __name__ == '__main__':
    unittest.main()
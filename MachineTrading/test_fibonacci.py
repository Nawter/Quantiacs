import unittest

# Returns the nth Fibonacci number
# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233
def fib(n):
    if n == 0: return 0
    elif (n ==1): return 1
    else: return fib(n-1) + fib(n-2)

def fib_ret(n):
    a,b=0,1
    for i in range(n):
        a,b=b,a+b
    return a


class TestFibonacci(unittest.TestCase):
    def test_fib_zero(self):
        self.assertEqual(fib_ret(0),0)

    def test_fib_1(self):
        self.assertEqual(fib_ret(1),1)

    def test_fib_two(self):
        self.assertEqual(fib_ret(2), 1)

    def test_fib_1(self):
        self.assertEqual(fib_ret(1), 1)

    def test_fib_out_of_range(self):
            self.assertNotEqual(fib_ret(-1), 1)



if __name__ == '__main__':
    unittest.main()
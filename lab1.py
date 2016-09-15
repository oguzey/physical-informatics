

# 1. generation
def henon_generator(x0, x1, amount):
    b = 0.3
    a = 1.4

    x_n = x1
    x_n_minus_1 = x0
    for x in xrange(amount):
        y_n = b * x_n_minus_1
        x_n_plus_1 = 1 - a * (x_n * x_n) + y_n
        x_n_minus_1 = x_n
        x_n = x_n_plus_1
        print x_n_plus_1
        yield x_n_plus_1


def calc_block_number(sequence):
    """
    calculate Sn(V)
    """
    max_pow = len(sequence)
    res = 0
    for x in xrange(max_pow):
        res += sequence[x] * 2 ** x
    return res

N = 10000
s = map(lambda x: x > 0, henon_generator(0.01, 0.1, N))

print s
block_sub_seq = {}

for v in xrange(2, 11):
    local_N = N / v
    print "V = %d, [N/v] = %d" % (v, local_N)
    block_sub_seq[v] = [s[(v * n):(v * (n + 1))] for n in xrange(local_N)]
    print block_sub_seq[v]
    for x in block_sub_seq[v]:
        print calc_block_number(x)
import math


def henon_generator(x0, x1, amount):
    """
    generation s sequence
    """
    b = 0.3
    a = 1.4

    x_n = x1
    x_n_minus_1 = x0
    for x in xrange(amount):
        y_n = b * x_n_minus_1
        x_n_plus_1 = 1 - a * (x_n * x_n) + y_n
        x_n_minus_1 = x_n
        x_n = x_n_plus_1
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


def calc_probability_blocks(blocks):
    count_all_blocks = float(len(blocks))
    counting = {}
    for block in blocks:
        counting[block] = counting.get(block, 0) + 1

    probabilities = map(lambda b: counting[b] / count_all_blocks, blocks)

    print 'probabilities = {}'.format(probabilities)
    return probabilities


N = 100
# generate s sequence
s = map(lambda x: x > 0, henon_generator(0.01, 0.1, N))

print s
block_sub_seq = {}

for v in xrange(2, 11):
    local_N = N / v
    print "V = %d, [N/v] = %d" % (v, local_N)
    # split s by part which has len = v
    block_sub_seq[v] = [s[(v * n):(v * (n + 1))] for n in xrange(local_N)]
    # convert bit blocks to numbers Sn(V)
    block_sub_seq[v] = map(calc_block_number, block_sub_seq[v])
    print 'block_sub_seq[v] = '
    print block_sub_seq[v]
    # calculate probabilities for chain blocks
    probs = calc_probability_blocks(block_sub_seq[v])
    print 'probs = '
    print probs

    shenon_entropy = - reduce(lambda acc, prob: acc + prob * math.log(prob, 2), probs, 0)
    print 'shenon entropy = ' + str(shenon_entropy)

import math
from functools import reduce
import matplotlib.pyplot as plt


def henon_generator(x0, x1, amount):
    """
    generation s sequence
    """
    b = 0.3
    a = 1.4

    x_n = x1
    x_n_minus_1 = x0
    for x in range(amount):
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
    for x in range(max_pow):
        res += sequence[x] * 2 ** x
    return res


def calc_probability_blocks(blocks):
    count_all_blocks = float(len(blocks))
    counting = {}
    for block in blocks:
        counting[block] = counting.get(block, 0) + 1

    probabilities = []
    for block, count in counting.items():
        probabilities.append(count / count_all_blocks)

    return probabilities

N = 10000
# generate s sequence
s = list(map(lambda x: x > 0, henon_generator(0.51, 0.5, N)))

vs = [x for x in range(2, 11)]
entropies = {'shenon': [],
             'renyis': {2: [], 3: []},
             'engine': {'shenon': [],
                        'renyis': {2: [], 3: []}}}

for v in vs:
    local_N = int(N / v)
    print("V = %d, [N/v] = %d" % (v, local_N))
    # split s by part which has len = v
    v_grams = [s[(v * n):(v * (n + 1))] for n in range(local_N)]
    # convert bit blocks to numbers Sn(V)
    v_grams = list(map(calc_block_number, v_grams))
    # calculate probabilities for chain blocks
    probs = calc_probability_blocks(v_grams)

    shenon_entropy = - reduce(lambda acc, prob: acc + prob * math.log(prob, 2), probs, 0)
    print('Block shenon entropy = ' + str(shenon_entropy))
    entropies['shenon'].append(shenon_entropy)
    for B in [2, 3]:
        renyi_entropy = - (1.0 / (B - 1)) * math.log(reduce(lambda acc, prob: acc + prob ** B, probs, 0), 2)
        print('renyi_entropy with B {} equal to {}'.format(B, renyi_entropy))
        entropies['renyis'][B].append(renyi_entropy)


diff_vs = [v * 10 + v + 1 for v in range(2, 10)]
for index in range(len(entropies['shenon']) - 1):
    entropies['engine']['shenon'].append(entropies['shenon'][index + 1] - entropies['shenon'][index])
    for B in [2, 3]:
        entropies['engine']['renyis'][B].append(entropies['renyis'][B][index + 1] - entropies['renyis'][B][index])


def make_figure(name):
    fig = plt.figure()
    fig.set_size_inches(15, 7)
    fig.canvas.set_window_title(name.capitalize())


make_figure('Graphs of entropies')
plt.title('Entropies')
plt.plot(vs, entropies['shenon'], 'o-', label='Shenon')
plt.plot(vs, entropies['renyis'][2], 'o-', label='Renyi2')
plt.plot(vs, entropies['renyis'][3], 'o-', label='Renyi3')
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend(loc=1, prop={'size': 10})

make_figure('Graphs of engine')
plt.title('Entropies of Engine')
plt.plot(diff_vs, entropies['engine']['shenon'], 'o-', label='Engine by Shenon')
plt.plot(diff_vs, entropies['engine']['renyis'][2], 'o-', label='Engine by Renyi2')
plt.plot(diff_vs, entropies['engine']['renyis'][3], 'o-', label='Engine by Renyi3')
plt.grid(b=True, which='major', color='grey', linestyle='--')
plt.legend(loc=1, prop={'size': 10})

plt.show()


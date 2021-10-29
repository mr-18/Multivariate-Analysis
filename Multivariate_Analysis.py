'''
Y3 Particle Physics Computing
Multivariate Analysis Project Code
'''
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import gauss as r_gauss
from numpy.random import poisson as r_poiss
##########Choosing number of decimal places for accuracy of events############
dp = 0
##############################################################################
class Distribution():
    '''
    Class representing a toy distribution of background and signal events in
    the same variable
    '''
    def __init__(self):
        '''
        Initialises the class
        '''
        pass
    def plot(self):
        '''
        Plots the sample values against the number of events
        '''
        count, background, signal = self.sample()
        fig = plt.figure(figsize=(8, 8))
        plt.title('Distribution of values against its corresponding number of events')
        plt.scatter(count.keys(), count.values(), label='Total events')
        # plt.scatter(signal.keys(), signal.values(), label='Signal events')
        # plt.scatter(background.keys(), background.values(), label='Background events')
        plt.xlabel('X value (Invariant mass)')
        plt.ylabel('Events')
        plt.legend()
        plt.show()
    def sample(self):
        '''
        Creates the sample values for the toy distribution.

        Returns
        -------
        count : dict
            Distribution of both background and signal events.
        background_dist : dict
            Distribution of only background events.
        signal_dist : dict
            Distribution of only signal events.

        '''
        n = 10000
        values = []
        count = {}
        background_dist = {}
        signal_dist = {}
        pbar = tqdm(total=n*0.52)    #progress bar total proportional to average number of values created each run
        while len(values) < n:
            background = r_gauss(10, 6)
            rounded_b = round(background, dp)
            signal = r_gauss(15, 5)
            rounded_s = round(signal, dp)
            if 0 < background < 22:
                count[rounded_b] = count.get(rounded_b, 0) + 1
                background_dist[rounded_b] = background_dist.get(rounded_b, 0) + 1
                values.append(background)
            if 5 < signal < 25:
                count[rounded_s] = count.get(rounded_s, 0) + 1
                signal_dist[rounded_s] = signal_dist.get(rounded_s, 0) + 1
                values.append(signal)
            pbar.update(1)
        pbar.close()
        return count, background_dist, signal_dist
    
def Optimisation(x, y, z, low_bounds=None):
    '''
    Calculates the significance of the sample distribution for different low
    bounds. Iterating through values from the minimum value of the keys in the
    signal distribution to the maximum value of the keys in the background
    distribution.

    Parameters
    ----------
    x : dict
        Distribution of both background and signal events.
    y : dict
        Distribution of only background events.
    z : dict
        Distribution of only signal events.
    low_bounds : int or float
        Specific low bound value to calculate significance for(used in Nelder-Mead).
        The default is None.

    Returns
    -------
    low_bounds : array, int or float
        Values for the low bound in the corresponding significance calculation.
    S_array : array, int or float
        Values for significance found with corresponding low bounds.

    '''
    #Option to run Significance calculation for input values with NM method
    if low_bounds:
        if isinstance(low_bounds, (float, int)):
            N_b = 0
            N_s = 0
            for n in z.keys():
                if n > low_bounds:
                    N_s += z[n]
            for n in y.keys():
                if n > low_bounds:
                    N_b += y[n]
            S_array = N_s/(N_b**(1/2))
        else:
            S_array = []
            for x in low_bounds:
                N_b = 0
                N_s = 0
                for n in z.keys():
                    if n > x:
                        N_s += z[n]
                for n in y.keys():
                    if n > x:
                        N_b += y[n]
                S = N_s/(N_b**(1/2))
                S_array.append(S)
    else:
        low_bounds = []
        S_array = []
        for low_bound in range(int(min(z.keys())), int(max(y.keys()))):
            N_b = 0
            N_s = 0
            low_bounds.append(low_bound)
            for n in z.keys():
                if n > low_bound:
                    N_s += z[n]
            for n in y.keys():
                if n > low_bound:
                    N_b += y[n]
            S = N_s/(N_b**(1/2))
            S_array.append(S)
    return low_bounds, S_array


def Single_Bisection(a, b, low_bounds, S_values):
    '''
    Executes the bisection method once; an optimisation algorithm for a single
    dimension

    Parameters
    ----------
    a : float/integer
        lower bound for interval bisection.
    b : float/integer
        upper bound for interval bisection.
    low_bounds : array
        Values for the low bound in the corresponding significance calculation.
    S_values : array
        Values for significance found with increasing low bounds.

    Returns
    -------
    m : float
        midpoint of interval for current iteration of bisection
    l : float
        midpoint of left interval for current iteration of bisection
    r : float
        midpoint of right interval for current iteration of bisection

    '''
    try:
        m = (a + b)/2
        l = (a + m)/2
        r = (m + b)/2
    except KeyError:
        print('bound out of sample range (5 <= (a,b) <= 25)')
    a_index = int(a-min(low_bounds))
    b_index = int(b-min(low_bounds))
    m_index = int(m-min(low_bounds))
    l_index = int(l-min(low_bounds))
    r_index = int(r-min(low_bounds))
    plt.title('Interval Bisection in progress')
    plt.scatter(low_bounds, S_values)
    plt.xlim(min(low_bounds)-2, max(low_bounds)+2)
    plt.ylim(min(S_values)-5, max(S_values)+5)
    plt.axvline(a, ymin=0, ymax=S_values[a_index], label='left bound', color='red')
    plt.axvline(b, ymin=0, ymax=S_values[b_index], label='right bound', color='orange')
    plt.axvline(m, ymin=0, ymax=S_values[m_index], label='midpoint', color='black', linestyle='-')
    plt.axvline(l, ymin=0, ymax=S_values[l_index], label='left interval midpoint', color='black', linestyle='--')
    plt.axvline(r, ymin=0, ymax=S_values[r_index], label='right interval midpoint', color='black', linestyle='--')
    plt.legend()
    plt.xlabel('X value')
    plt.ylabel('Significance')
    if (b-a) != 0.:
        plt.draw()
        plt.pause(0.01)
        plt.clf()
    else:
        plt.close()
        plt.show()
    return m, l, r
def Bisection_Iteration(iterations=None):
    '''
    Executes the 'single_bisection' method for a number of iterations 
    or until b - a is small enough; an optimisation algorithm for a single
    dimension
    
    Parameters
    ----------
    iterations : integer
        number of times to perform interval bisection method.

    Returns
    -------
    a : float
        Highest value of significance.

    '''
    x, y, z = Distribution().sample()
    low_bounds, S_values = Optimisation(x, y, z)
    fig = plt.figure(figsize=(8, 8))
    plt.ion()
    N = 0
    if iterations:
        while N < iterations:
            if N == 0:
                a, b = min(low_bounds), max(low_bounds)
                m, l, r = Single_Bisection(a, b, low_bounds, S_values)
                a_index = int(a-min(low_bounds))
                b_index = int(b-min(low_bounds))
                m_index = int(m-min(low_bounds))
                l_index = int(l-min(low_bounds))
                r_index = int(r-min(low_bounds))
                f_a, f_b, f_m, f_l, f_r = S_values[a_index], S_values[b_index], S_values[m_index], S_values[l_index], S_values[r_index]
                function_array = [f_a, f_b, f_m, f_l, f_r]
                if max(function_array) == f_a:
                    b = m
                elif max(function_array) == f_l:
                    b = m
                elif max(function_array) == f_b:
                    a = m
                elif max(function_array) == f_r:
                    a = m
                elif max(function_array) == f_m:
                    a = l
                    b = r
            else:
                m, l, r = Single_Bisection(a, b, low_bounds, S_values)
                a_index = int(a-min(low_bounds))
                b_index = int(b-min(low_bounds))
                m_index = int(m-min(low_bounds))
                l_index = int(l-min(low_bounds))
                r_index = int(r-min(low_bounds))
                f_a, f_b, f_m, f_l, f_r = S_values[a_index], S_values[b_index], S_values[m_index], S_values[l_index], S_values[r_index]
                function_array = [f_a, f_b, f_m, f_l, f_r]
                if max(function_array) == f_a:
                    b = m
                elif max(function_array) == f_l:
                    b = m
                elif max(function_array) == f_b:
                    a = m
                elif max(function_array) == f_r:
                    a = m
                elif max(function_array) == f_m:
                    a = l
                    b = r
            N += 1
    else:
        a, b = min(low_bounds), max(low_bounds)
        while (b - a) != 0.:
            m, l, r = Single_Bisection(a, b, low_bounds, S_values)
            a_index = int(a-min(low_bounds))
            b_index = int(b-min(low_bounds))
            m_index = int(m-min(low_bounds))
            l_index = int(l-min(low_bounds))
            r_index = int(r-min(low_bounds))
            f_a, f_b, f_m, f_l, f_r = S_values[a_index], S_values[b_index], S_values[m_index], S_values[l_index], S_values[r_index]
            function_array = [f_a, f_b, f_m, f_l, f_r]
            if max(function_array) == f_a:
                b = m
            elif max(function_array) == f_l:
                b = m
            elif max(function_array) == f_b:
                a = m
            elif max(function_array) == f_r:
                a = m
            elif max(function_array) == f_m:
                a = l
                b = r
            N += 1
    print('\nLargest significance of value {} found to occur at an invariant mass value of {} after {} iterations'.format(S_values[a_index], a, N))
    return a

def Optimised_significance_plot(iterations, bins=20):
    '''
    Calculates the significance of each toy experiment after implementing
    the optimum cut value found in the `Optimisation` function

    Parameters
    ----------
    iterations : int
        the number of toy experiments to calculate significance for.
    bins : int
        the number of bins for the significance values to be split into when plotted.

    Returns
    -------
    new_distribution : dict
        Distribution of both background and signal events after implementing poisson distribution.
    new_background : dict
        Distribution of background events after implementing poisson distribution.
    new_signal : dict
        Distribution of signal events after implementing poisson distribution.
    S_values : array
        Significance values calculated for each toy experiment.

    '''
    count, background, signal = Distribution().sample()
    Optimised_low_cut_value = float(Bisection_Iteration())
    S_values = []
    N = 0
    pbar = tqdm(total=iterations)
    while N <= iterations:
        new_distribution = {}
        new_signal = {}
        new_background = {}
        for i, j in background.items():
            if i >= Optimised_low_cut_value:
                lam = j**(1/2)
                new_background[i] = j + r_poiss(lam)
        for i, j in signal.items():
            if i >= Optimised_low_cut_value:
                lam = j**(1/2)
                new_signal[i] = j + r_poiss(lam)
        for i in new_signal.keys():
            for j in new_background.keys():
                if i == j:
                    new_distribution[i] = new_signal[i] + new_background[j]
                elif i > 22:
                    new_distribution[i] = new_signal[i]
        N_b = 0
        N_s = 0            
        for n in new_signal.keys():
            if n >= Optimised_low_cut_value:
                N_s += new_signal[n]
        for n in new_background.keys():
            if n >= Optimised_low_cut_value:
                N_b += new_background[n]
        S = N_s/(N_b**(1/2))
        S_values.append(S)
        N += 1
        pbar.update(1)
    pbar.close()
    fig = plt.figure(figsize=(8, 8))
    # plt.title('Plot showing the toy distributions with new poisson fluctuation and optimal low cut value of {}'.format(Optimised_low_cut_value))
    # plt.scatter(new_distribution.keys(), new_distribution.values(), label='Total events')
    # plt.scatter(new_signal.keys(), new_signal.values(), label='Signal events')
    # plt.scatter(new_background.keys(), new_background.values(), label='Background events')
    # plt.xlabel('X value')
    # plt.ylabel('Count')
    # plt.legend()
    plt.title('Plot showing the Significance from {} different experiments'.format(iterations))
    plt.hist(S_values, bins=bins)
    plt.xlabel('Significance')
    plt.ylabel('Count')
    plt.show
    return new_distribution, new_background, new_signal, S_values
def Nelder_Mead(no_improv_threshold=10e-6, no_improv_break=10, max_iterations=0, alpha = 1, gamma = 2, rho = 0.5, sigma = 0.5):
    a, b, c = Distribution().sample()
    x_values, S_values = Optimisation(a, b, c)
    #Nelder-Mead minimises so taking negative of optimisation function to maximise it
    x_init = []
    for x in x_values:
        # x_i = -1 * x
        x_init.append(x)
    S_init = []
    for S in S_values:
        S_i = -1 * S
        S_init.append(S_i)
    dim = len(x_init)
    no_improv = 0
    xS_array = []
    x = []
    for i in range(dim):
        x = x_init.copy()
        score = Optimisation(a, b, c, x[i])
        xS_array.append((score[0], -score[1]))
    #simplex iteration
    iterations = 0
    while 1:
        #Order
        N = 0
        while N < dim-1:
            for i in range(dim-1):
                if xS_array[i][1] > xS_array[i+1][1]:
                    k, l = x[i], x[i+1]
                    m, n = xS_array[i], xS_array[i+1]
                    x[i] = l
                    x[i+1] = k
                    xS_array[i] = n
                    xS_array[i+1] = m
            N += 1
        best = xS_array[0][1]
        #break after max iterations
        if max_iterations and iterations >= max_iterations:
            return xS_array[0]
        iterations += 1
        #break after no_improv_break iterations with no improvement
        print('Optimal low cut value so far: {}'.format(best))
        if best < min(S_init) - no_improv_threshold:
            no_improv = 0
            S_init = best
        else:
            no_improv += 1
        if no_improv >= no_improv_break:
            return xS_array[0]
        #centroid
        x_0 = 0
        for tup in xS_array[:-1]:
            x_0 += tup[1] / (len(xS_array)-1)
        
        #reflection
        x_r = x_0 + alpha*(x_0 - xS_array[-1][1])
        r_score = Optimisation(a, b, c, x_r)
        print(r_score)
        if xS_array[0][1] <= r_score < xS_array[-2][1]:
            del xS_array[-1]
            xS_array.append([r_score])
            continue
        #expansion
        if r_score > xS_array[0][1]:
            x_e = x_0 + gamma*(x_0 - xS_array[-1][0])
            e_score = Optimisation(a, b ,c, x_e)
            if e_score < r_score:
                del xS_array[-1]
                xS_array.append([e_score])
                continue
            else:
                del xS_array[-1]
                xS_array.append([r_score])
                continue
        #contraction
        x_c = x_0 + rho*(x_0 - xS_array[-1][0])
        c_score = Optimisation(a, b, c, x_c)
        if c_score < xS_array[-1][1]:
            del xS_array[-1]
            xS_array.append([c_score])
            continue
        #reduction
        x_1 = xS_array[0][0]
        new_xS = []
        for tup in xS_array:
            redx = x_1 + sigma*(tup[0] - x_1)
            score = Optimisation(a, b, c, redx)
            new_xS.append([score])
        xS_array = new_xS
    return xS_array[0]
if __name__=='__main__':
    # a = Distribution().plot()
    # Bisection_Iteration()
    Optimised_significance_plot(1000)
    # Nelder_Mead()
    
"""This file contains all of the helper methods used in the ELFI model. Many of them
are wrappers transforming data to a form that is more efficient to save or easier to
handle in ELFI.
"""

import simulator as si
import numpy as np
import scipy.stats as ss
import warnings


def Rt_to_d(R, t):
    """Compute death rates from the reproductive value and net transmission rate

    Parameters
    ----------
    R : float, np.ndarray
        Reproductive value
    t : float, np.ndarray
        Net transmission rate

    Returns
    -------
    float, np.ndarray
        Death rate
    """
    return t/np.abs(R-1)


def Rt_to_a(R, t):
    """Compute birth rates from the reproductive value and net transmission rate

    Parameters
    ----------
    R : float, np.ndarray
        Reproductive value
    t : float, np.ndarray
        Net transmission rate

    Returns
    -------
    float, np.ndarray
        Birth rate(s)
    """
    return t + Rt_to_d(R, t)


def analytic_R0_upper_bound(R1, p_compliance):
    """Compute the upper bound for R0 ensuring that the population size balance value exists.

    Parameters
    ----------
    R1 : float, np.ndarray
        Reproductive value of the non-compliant population
    p_compliance : float

    Returns
    -------
    float, np.ndarray
        The non-inclusive upper bound(s)
    """
    p_non_c = 1 - p_compliance
    return (1 - R1 * p_non_c)/p_compliance


class JointPrior:
    @staticmethod
    def rvs(burden, mean_obs_bounds=None, t1_bound=100, a1_bound=300, p_compliance=.95, R1_bound=None, size=None,
            random_state=None):
        """The joint prior for the San Francisco data

        Parameters
        ----------
        burden : float, np.array
        mean_obs_bounds : list, tuple, optional
            Non-inclusive lower and upper bounds for the mean of observed hosts in one year.
        t1_bound : float
            Upper bound for the allowed net transmission rate of the non-compliant hosts (a1 - d1).
        a1_bound : float
            Upper bound for the transmission rate of the non-compliant hosts.
        p_compliance : float, optional
            Probability that a new host will be compliant to therapy. Default value is 0.95.
        R1_bound : float, optional
            Will be set automatically to 1/(1 - p_compliance), which is the non-inclusive upper limit
            for stabile populations.
        size : int, optional
        random_state : np.random.RandomState, optional

        """

        p_non_c = 1 - p_compliance
        R1_bound = R1_bound or 1/p_non_c
        if not p_non_c*R1_bound <= 1:
            warnings.warn('R1_bound is too high for a stabile population.')

        random_state = random_state or np.random

        size_ = size or 1
        if isinstance(size_, tuple):
            if len(size_) > 1:
                raise ValueError('Size must be scalar')
            size_ = size_[0]

        burden = burden
        d0 = 5.95

        i = 0
        R1 = np.empty(size_)
        t1 = np.empty(size_)
        R0 = np.empty(size_)
        m_obs = np.empty(size_)

        while True:
            R1_ = random_state.uniform(1.01, R1_bound, size=size_)
            t1_ = random_state.uniform(0.01, t1_bound, size=size_)

            R0_bounds_ = analytic_R0_upper_bound(R1_, p_compliance)
            R0_ = random_state.uniform(0.01, 1, size=size_)*R0_bounds_

            a0_ = R0_*d0
            d1_ = Rt_to_d(R1_, t1_)
            a1_ = Rt_to_a(R1_, t1_)

            means_ = np.array(si.analytical_means(burden, a0_, d0, a1_, d1_)).T

            mask = np.ones(size_, dtype=bool)
            if mean_obs_bounds is not None:
                mask = np.logical_and(mask, means_[:, 2] < mean_obs_bounds[1])
                mask = np.logical_and(mask, means_[:, 2] > mean_obs_bounds[0])
            if a1_bound:
                mask = np.logical_and(mask, a1_ < a1_bound)

            sl = slice(i, min(i + np.sum(mask), size_))
            l = sl.stop - sl.start

            R0[sl] = R0_[mask][:l]
            R1[sl] = R1_[mask][:l]
            t1[sl] = t1_[mask][:l]
            m_obs[sl] = means_[mask, 2][:l]

            i = sl.stop

            if i == size_:
                break

        if size is None:
            R0 = R0[0]
            R1 = R1[0]
            t1 = t1[0]
            m_obs = m_obs[0]

        return np.array((R0, R1, t1, m_obs)).T


class DummyPrior:
    """A placeholder marginal distribution from the joint prior distribution

    """
    @staticmethod
    def rvs(joint_rvs, index, size, random_state):
        """

        Parameters
        ----------
        joint_rvs : np.ndarray
            Random variates from the joint prior
        index : int
            Index for choosing the specific variable from the joint prior
        size
        random_state

        Returns
        -------

        """
        return joint_rvs[:, index]


def pick(y, column):
    """A helper method to pick a specific output from the simulator node output.

    Allows one to create separate nodes for specific outputs, e.g. number of clusters.
    """
    return y[column]


def _get_n_clusters(observed):
    """Compute the number of clusters in the data produced directly by the simulator

    Parameters
    ----------
    observed : np.ndarray
        Vector of cluster sizes produced directly by the simulator

    """
    observed = np.atleast_1d(observed)
    clusters_ = np.atleast_2d(observed)
    n = np.sum(clusters_ > 0, 1)
    return n.item() if observed.ndim == 1 else n


def cluster_size_hist(observed, cluster_size_bound):
    """Compute a histogram of cluster sizes

    Parameters
    ----------
    observed : np.array
        Data directly from the simulator
    cluster_size_bound : int
        Largest allowed cluster size in the histogram (for saving storage)

    Returns
    -------

    """
    # Do not get multiple values to the last bin because the endpoint is included
    bins = list(range(cluster_size_bound + 1)) + [cluster_size_bound + 1.5]
    csh, _ = np.histogram(observed, bins=bins, )
    csh[0] = 0
    return csh.astype(dtype=observed.dtype), \
           np.sum(observed > cluster_size_bound, dtype=observed.dtype)


def obs_times_hist(obs_times, t_obs):
    """Return a histogram of observations per month

    Parameters
    ----------
    obs_times : np.ndarray
        Observation times from the simulator
    t_obs : int
        Observation period in years

    Returns
    -------

    """
    # Convert to months
    t_obs *= 12
    t = obs_times*12

    # Note that the dtype must match that produced by the simulator
    bins = np.arange(t_obs + 1, dtype=obs_times.dtype)
    counts, _ = np.histogram(t, bins=bins, range=(0, t_obs+1))

    return counts.astype('i2')


def get_SF_data(cluster_size_bound, dtype=np.int16):
    """Get San Francisco Bay area Tuberculosis epidemic data.

    The data is formatted to work with the provided ELFI model.
    """
    clusters_list = []
    for c_size, c_num in [(30,1), (23, 1), (15,1), (10,1), (8,1), (5,2), (4,4), (3,13), (2,20), (1,282)]:
        clusters_list += [c_size]*c_num

    clusters = np.array(clusters_list, dtype=dtype)

    obs_times = np.ones(cluster_size_bound, dtype=np.float16)*np.nan
    obs_times[:30] = np.array([0,0,1,2,3,3,4,4,4,4,4,4,5,5,5,5,5,7,8,8,10,11,12,14,16,16,18,20,21,23])/12

    # True values for checking
    n_obs = 473
    n_clusters = 326

    # Test some known statistics
    assert sum(clusters) == n_obs
    assert _get_n_clusters(clusters) == n_clusters

    # Test that those are preserved after transform
    csh, n_o = cluster_size_hist(clusters, cluster_size_bound)
    assert len(csh) == cluster_size_bound + 1
    assert sum(csh) == n_clusters
    assert np.sum(np.arange(len(csh))*csh) == n_obs

    oth = obs_times_hist(obs_times, 2)

    y = np.array([(csh, n_obs, n_clusters, np.max(clusters), oth, n_o, 0)],
                 dtype=[('clusters', dtype, cluster_size_bound+1),
                        ('n_obs', dtype),
                        ('n_clusters', dtype),
                        ('largest', dtype),
                        ('obs_times', dtype, 24),
                        ('n_oversized', dtype),  # TODO: allow ELFI to run without these two
                        ('time', np.float16)])

    return y


def simulator(burden, a0, d0, a1, d1, t_obs, cluster_size_bound, warmup_bounds=(0, 300), batch_size=1,
              random_state=None, dtype=np.int16):
    """Wrapper for the simulator to make it compatible with the ELFI model

    This adds support for batch_size, transforms the data to a form that is efficient to store
    etc.

    Parameters
    ----------
    burden
    a0
    d0
    a1
    d1
    t_obs
    cluster_size_bound : int
        Upper bound for the cluster sizes
    min_warmup : int
        Minimum years for the warmup.
    batch_size
    random_state
    dtype : np.dtype

    Notes
    -----
    Observed clusters will be truncated to fit clusters_bound if exceeded. Also obs_times will be
    truncated to cluster_size_bound.

    Returns
    -------
    np.ndarray

    """
    y = np.zeros(batch_size, dtype=[('clusters', dtype, cluster_size_bound+1),
                                    ('n_obs', dtype),
                                    ('n_clusters', dtype),
                                    ('largest', dtype),
                                    ('obs_times', dtype, t_obs*12),
                                    ('n_oversized', dtype),
                                    ('n_c', dtype),
                                    ('n_nc', dtype),
                                    ('time', np.float16),])

    min_warmup, max_warmup = warmup_bounds

    for i in range(batch_size):
        args = [a[i] if hasattr(a, '__len__') else a for a in (burden, a0, d0, a1, d1)]
        s = si.Simulator(*args, t_obs=t_obs, t_warmup=np.inf, random_state=random_state,
                         dtype=dtype)

        m_c, m_nc, m_obs = s.analytical_means

        # Determine warmup
        years = 0
        m_c_unreached = True
        m_nc_unreached = True
        while years < max_warmup/2 and (m_c_unreached or m_nc_unreached):
            years += 1
            s.advance_to(years)

            if s.n_c >= m_c:
                m_c_unreached = False
            if s.n_nc >= m_nc:
                m_nc_unreached = False

        # Set warmup
        s.t_warmup = max(2*years, min_warmup)

        # Run the rest of the warmup and the observation time
        while s.next_event():
            pass

        y[i]['clusters'], n_o = cluster_size_hist(s.observed, cluster_size_bound)
        y[i]['n_obs'] = s.n_obs
        y[i]['n_clusters'] = _get_n_clusters(s.observed)
        y[i]['largest'] = np.max(s.observed)
        y[i]['obs_times'] = obs_times_hist(s.extract_death_times(dtype=np.float16), t_obs)
        y[i]['n_oversized'] = n_o
        y[i]['n_c'] = s.n_c
        y[i]['n_nc'] = s.n_nc
        y[i]['time'] = s.time

    return y


"""Utils"""


# First and last observation month
def obs_period(times):
    times = np.atleast_2d(times)
    mask = times > 0
    a = np.argmax(mask, 1)
    b = times.shape[1] - np.argmax(np.flip(mask, 1), 1)
    return b-a


def get_largest_cluster_sizes(clusters, n=4):
    clusters = clusters.copy()
    largest = np.zeros((len(clusters), n))
    for i in range(n):
        cluster_sizes = (clusters > 0) * np.arange(clusters.shape[1])
        current_largest = np.max(cluster_sizes, 1)

        # Mark it as used
        clusters[:, current_largest] -= 1

        largest[:, i] = current_largest

    return largest


def mean_largest_diff(clusters):
    sizes = get_largest_cluster_sizes(clusters, 4)
    return np.mean(sizes[:, :-1] - sizes[:, 1:], 1)


"""Distances"""


def distance(n_obs, n_clusters, largest, clusters, obs_times=None, observed=None):
    """

    Parameters
    ----------
    n_obs
    n_clusters
    largest
    clusters
    obs_times
    observed

    Returns
    -------

    """
    d0 = n_obs - observed[0]
    d1 = n_clusters - observed[1]

    d_largest = largest - observed[2]
    d_largest_diff = mean_largest_diff(clusters) - mean_largest_diff(observed[3])

    # Compare the number of small clusters by relative error
    c_sizes = slice(1, 3)
    true_relative_size = observed[3][:, c_sizes] / observed[0]
    dc = clusters[:, c_sizes] / n_obs[:, None] - observed[3][:, c_sizes] / observed[0]

    # Relative error
    dc = dc / true_relative_size

    # Observation times are optional
    if obs_times is not None:
        d_op = obs_period(obs_times) - obs_period(observed[4])
        d_om = np.sum(obs_times > 0, 1) - np.sum(observed[4] > 0, 1)
    else:
        d_op = np.zeros(len(largest))
        d_om = np.zeros(len(largest))

    d = np.linalg.norm(np.c_[d0, d1, 2 * d_largest, 10 * d_largest_diff, 100 * dc, 10 * d_op, 10 * d_om],
                       ord=2, axis=1)

    return d

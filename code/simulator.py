"""Communicable disease outbreak simulator"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Simulator:
    """

    Attributes
    ----------

    time : float
        Time passed so far in years

    """

    def __init__(self, burden, trans, death, trans_non_c, death_non_c, p_compliance=.95,
                 p_observed=.80, t_obs=2, t_warmup=None, random_state=None,
                 track_clusters=True, dtype=np.int32):
        """

        Parameters
        ----------
        burden : float
            Rate of new active TB cases appearing as a result of reactivations and
            immigration. These will all have a distinct mutation of the pathogen.
        trans : float
            Transmission rate for the compliant population
        death : float
        trans_non_c : float
            Transmission rate for the non-compliant cases
        death_non_c : float
            Death rate for the non-compliant cases
        p_compliance : float
            Probability that a new active TB case will be compliant with therapy
        p_observed : float
            Probability that the case becomes observed at "death" event
        t_obs : float, optional
            The observation time in years for the self.observed data. Default is 2 years
        t_warmup : float, optional
            The time to run the process before starting the observation period. Default is
            t_obs.
        random_state : np.random.RandomState, optional
        track_clusters : bool, optional
            Whether to track clusters in the population. Works slower if True (default).
        dtype : np.dtype, optional
            dtype of the cluster vectors and population size integers.

        Returns
        -------

        """

        self.t_obs = t_obs
        self.t_warmup = t_warmup if t_warmup is not None else self.t_obs
        self.random_state = random_state or np.random
        self.p_compliance = p_compliance

        self.p_observed = p_observed
        self.track_clusters = track_clusters

        # State variables
        self.time = 0
        self.n_events = 0

        # Cluster book keeping, non_compliant, compliant and observed respectively
        self.clusters = np.zeros((3, 0), dtype=dtype)

        # Population sizes book keeping as above
        self.n = np.zeros(3, dtype=np.dtype)

        # Number of clusters that have ever existed
        self.n_clusters = 0

        # Death times book keeping only for clusters of size > 1
        self.death_times = {}
        self._free_slots = []

        self.base_rates = np.array([burden, trans, death, trans_non_c, death_non_c])

    @property
    def burden(self):
        return self.base_rates[0]

    @property
    def trans(self):
        return self.base_rates[1]

    @property
    def death(self):
        return self.base_rates[2]

    @property
    def trans_non_c(self):
        return self.base_rates[3]

    @property
    def death_non_c(self):
        return self.base_rates[4]

    @property
    def n_nc(self):
        return self.n[0]

    @property
    def n_c(self):
        return self.n[1]

    @property
    def n_obs(self):
        return self.n[2]

    @property
    def non_compliant(self):
        return self.clusters[0]

    @property
    def compliant(self):
        return self.clusters[1]

    @property
    def observed(self):
        return self.clusters[2]

    def advance_to(self, time=1):
        """Advance the simulator to time `time` in years.

        New events are drawn until we reach or go above the parameter `time`.

        Parameters
        ----------

        time : float, optional
            Time in years to advance to. Default is one year.

        Notes
        -----
        The simulation is stopped immediately when the observation time has been passed even
        if the time has not been reached.

        Returns
        -------
        bool
            Returns False when the simulation has stopped.
        """
        cont = True
        while self.time < time and cont:
            cont = self.next_event()
        return cont

    def next_event(self):
        """Draw the next event.

        Notes
        -----
        The next event will not be simulated if it goes beyond the observation time.

        Returns
        -------
        bool
            Returns False when the simulation has stopped.

        """

        n_c = self.n[1]
        n_nc = self.n[0]
        rates = self.base_rates * (1, n_c, n_c, n_nc, n_nc)
        sum_rates = np.sum(rates)

        # Draw the time to the next event ...
        dt = self.random_state.exponential(scale=1/sum_rates)
        if self.time + dt >= self.t_warmup + self.t_obs:
            return False

        # ... and advance the time
        self.time += dt
        self.n_events += 1

        # Which event
        e = self.random_state.multinomial(1, rates / sum_rates)

        # New TB cluster appears
        if e[0]:
            self._add_cluster()

        # Transmission from compliant hosts
        elif e[1]:
            self._add_transmission(self.n[1], self.clusters[1])

        # Death from compliant hosts
        elif e[2]:
            self._add_death(1)

        # Transmission from non-compliant hosts
        elif e[3]:
            self._add_transmission(self.n[0], self.clusters[0])

        # Death from non-compliant hosts
        elif e[4]:
            self._add_death(0)

        else:
            raise ValueError('Something went wrong')

        return True

    def _draw_compliance(self):
        return int(self.random_state.rand() <= self.p_compliance)

    def _add_cluster(self):
        self.n_clusters += 1

        compliant = self._draw_compliance()
        self.n[compliant] += 1

        if not self.track_clusters:
            return

        slot = self._get_slot()
        self.clusters[compliant, slot] = 1

    def _add_transmission(self, n, clusters):
        """

        Parameters
        ----------
        n : int
            Population size of clusters
        clusters
            The cluster array from where the transmission originated
        """
        compliant = self._draw_compliance()
        self.n[compliant] += 1

        if not self.track_clusters:
            return

        i = _choose_cluster(clusters, n, self.random_state)
        self.clusters[compliant, i] += 1

    def _add_death(self, compliant):
        self.n[compliant] -= 1

        observe = (self.time >= self.t_warmup) and (self.random_state.rand() <= self.p_observed)
        if observe:
            self.n[2] += 1

        if not self.track_clusters:
            return

        # Record the death to the cluster
        i = _choose_cluster(self.clusters[compliant], self.n[compliant] + 1, self.random_state)
        self.clusters[compliant, i] -= 1

        if observe:
            self.clusters[2, i] += 1
            if i in self.death_times:
                self.death_times[i].append(self.time - self.t_warmup)
            else:
                self.death_times[i] = [self.time - self.t_warmup]

    def _extend_clusters(self):
        """Extend the cluster np.array"""
        w = self.clusters.shape[1]

        m = np.max(self.analytical_means[:2])
        # Extend by a minimum of 10
        m = int(np.max((10, m / 4, .2*w)))

        self.clusters = np.concatenate((self.clusters, np.zeros((3, m))), 1)

        return list(range(w, w + m))

    def _get_slot(self):
        if not self._free_slots:
            # A slot is free if no population has any individuals left in the cluster
            # and it has not been observed.
            self._free_slots = list(np.flatnonzero(np.all(self.clusters == 0, axis=0)))
            if not self._free_slots:
                # Make new slots
                self._free_slots = self._extend_clusters()

        return self._free_slots.pop()

    def extract_death_times(self, dtype=None):
        """Find the largest cluster and return it's death times"""
        observed = self.clusters[2]
        i_max = np.argmax(observed)
        return np.array(self.death_times[i_max], dtype=dtype)

    @property
    def analytical_means(self):
        """Return the balance population sizes and the approximated mean number of observations in one year

        Returns
        -------
        list
        """
        return analytical_means(self.burden, self.trans, self.death, self.trans_non_c,
                                self.death_non_c, self.p_compliance, self.p_observed)


def _choose_cluster(clusters, sum_clusters, random_state):
    p = clusters/sum_clusters
    return np.flatnonzero(random_state.multinomial(1, p)).item()


def analytical_means(burden, trans, death, trans_non_c, death_non_c, p_compliance=.95,
                     p_observed=.80):
    """Return the balance population sizes and the approximated mean number of observations in one year

    Returns
    -------
    list
    """

    p_c_neg = 1 - p_compliance

    # Compute the mean of the non-compliant population
    n_nc = burden * death * p_c_neg
    n_nc /= (death * death_non_c -
             trans * death_non_c * p_compliance -
             trans_non_c * death * p_c_neg)

    # Compute the mean of the compliant population
    n_c = n_nc * (death_non_c - trans_non_c * p_c_neg) - burden * p_c_neg
    n_c /= (trans * p_c_neg)

    # Compute the mean of observed in one year
    n_obs = (n_c * death + n_nc * death_non_c) * p_observed

    return n_c, n_nc, n_obs


def test_simulator():
    """A quick test sanity check for the simulator."""
    s = Simulator(200, 1, 6, trans_non_c=1.1, death_non_c=.52, t_obs=2, t_warmup=10)
    while s.next_event():
        pass

    assert np.array_equal(s.n, np.sum(s.clusters, 1))
    assert np.all(s.clusters >= 0)

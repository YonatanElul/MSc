import numpy as np
import DeterministicAnalytics.EntropyMethods.Utilities.CalcEnt as CalcEnt


class EKexp(object):

    """
    This is the implementation of the experiment of the AF prediction project, for a single patient, for all parameters.
    The right way to use it is to construct an occurrence for a patient, and then to use the run_exp() for all desired
    sets of parameters. If parallel computing is used, the different workers should each handle a patient and not a set
    of parameters.
    """

    def __init__(self, af_times: np.ndarray, r_times: np.ndarray = None, yyEnt_vec: np.ndarray = None,
                 yyTime: np.ndarray = None, tau: int = 1, order: int = 2, window_length: int = 60):
        """
        The constructor for a handler of the experiment for a patient. The constructor is somewhat versatile and can
        get the data or as a RR interval vector or explicitly as a pair of entropies vector  and a corresponding time
        vector.

        :param af_times: vector of the times of the AF events
        :param r_times: - optional - heart beat times vector
        :param yyEnt_vec: - optional - entropies vector
        :param yyTime: - optional - corresponding time vector for the entropy vector
        :param tau: This value determines the scale of the entropy moment to be computed. The default
        value is 1. Please note that: max{tau} = length(rr_peaks) - 1, with order = 1.
        :param order:int - Optional - This value determines the order of the entropy 'derivative' to be computed.
        Please note that: max{order} = length(rr_peaks) - 1, with tau = 1.
        :param window_length:int - optional - the length of the window for the calculation of the variance of the
        entropy.
        Please note that: max{window_length} = length(rr_peaks) - 3, but it better be window_length << length(rr_peaks)
        """

        if r_times is None and yyEnt_vec is None:
            raise ValueError('No input')

        if r_times is not None and yyEnt_vec is not None:
            raise ValueError('too many inputs')

        if yyEnt_vec is not None and yyTime is None:
            raise ValueError('time vector missing')

        if r_times is not None and yyTime is not None:
            raise ValueError('no need for a time vector when using beat times, argument yyTime is should be used for'
                             ' pre-calculated entropy.')

        self.var_vec = None
        if r_times is not None:
            self.var_vec, self.time = sample_runner(r_times, window_length, order, tau)

        if yyEnt_vec is not None:
            self.var_vec = local_variance(yyEnt_vec, window_length)
            self.time = yyTime[window_length:]

        if self.var_vec is None:
            raise ValueError('The variance is None - There is a problem with the inputs')

        self.af_times = af_times

    def run_exp(self, alpha: float, prediction_interval: float, v0: float, running_avg_window_len: int, n: int):
        """
        This method is used to run a single experiment on a single patient for a specific set of parameters.

        :param alpha: scalar, Level of certainty desired
        :param prediction_interval: scalar, defines how long before an AF to look for
        :param v0: scalar, baseline variance to compare with
        :param running_avg_window_len: scalar, size of window for the running average
        :param n:number of consecutive alarms to consider as a meaningful alarm
        :return fp: number of false positives
        :return fn: number of false negatives
        :return num_of_alarms: total number of alarms
        :return num_of_af_episodes: total number of AF episodes
        """

        running_avg_window = np.ones(int(running_avg_window_len))
        means = np.convolve(self.var_vec, running_avg_window, 'same')

        xt = []
        for i, val in enumerate(means):
            if i >= len(self.time):
                break

            if val < (alpha * v0):
                xt.append(self.time[i])

        alarms = []
        for i, t in enumerate(xt):
            if i < (n - 1):
                continue

            if (t - xt[i - n + 1]) < prediction_interval:
                alarms.append(t)

        events = []
        for i, t in enumerate(self.af_times):
            if i == 0:
                events.append(t)
                continue

            if (t - self.af_times[i - 1]) < prediction_interval:
                continue

            events.append(t)

        if not alarms:
            num_of_alarms = 0
            num_of_af_episodes = len(events)
            fp = 0
            fn = num_of_af_episodes

            return fp, fn, num_of_alarms, num_of_af_episodes

        fp = 0
        for i, alarm_time in enumerate(alarms):
            is_false = True

            if (i + 1) < len(alarms):
                if (alarms[i + 1] - alarm_time) < prediction_interval:
                    continue

            for event_time in events:
                if 0 < (event_time - alarm_time) < prediction_interval:
                    is_false = False
                    break

            if is_false:
                fp += 1

        fn = 0
        for i, event_time in enumerate(events):
            is_false = True

            for alarm_time in alarms:
                if 0 < (event_time - alarm_time) < prediction_interval:
                    is_false = False
                    break

            if is_false:
                fn += 1

        num_of_alarms = len(alarms)
        num_of_af_episodes = len(events)

        return fp, fn, num_of_alarms, num_of_af_episodes


def sample_runner(r_times, window_length, order, tau):
    rr_vec = np.diff(r_times)
    calculator = CalcEnt.EntropyManipulator()
    calculator.insert_rr(rr_peaks=rr_vec)
    time, mevs = calculator.calc_multi_scale_entropy_moment(time=r_times[1:], rr_peaks=rr_vec, tau=tau, order=order)

    return time[window_length:], local_variance(mevs, window_length)


def local_variance(vec: np.ndarray, window_length: int):
    length = len(vec)
    res = np.zeros(length - window_length)

    for i in range(0, len(res)):
        res[i] = np.var(vec[i:(i + window_length)])

    return res

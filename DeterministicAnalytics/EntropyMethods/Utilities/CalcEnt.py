import os
import pickle
import numpy as np


class EntropyManipulator(object):
    """
    This Class serves as the main object allowing manipulation and calculation of entropy related indices from a given
    RR-Beats interval point signal.
    """

    def __init__(self, rr_peaks: np.ndarray = None):
        """
        Description:
        The constructor for the EntropyManipulator class, which serves as the main tool for performing entropy-
        related calculations on a given RR-beats point signal

        :param rr_peaks: NumPy.ndaarray - Optional - Can be used as the signal on which the EntropyManipulator will
        work on. Please note that it is possible to initialize the EntropyManipulator with an input signal after the
        initiation of the class instance, using the 'insert_rr' method, which can also be used in order to replace the
        current signal on which the EntropyManipulator acts on.
        """

        self.rr_peaks = rr_peaks
        self.entropy = None
        self.mse = None

    def insert_rr(self, rr_peaks: np.ndarray):
        """
        Description:
        This method allows to insert an input signal for the EntropyManipulator to act on

        :param rr_peaks: NumPy.ndarray - The signal to be inserted
        :return: None
        """
        self.rr_peaks = rr_peaks

    def calc_multi_scale_entropy_moment(self, time: np.ndarray, rr_peaks: np.ndarray = None, tau: int = 1,
                                        order: int = 2):
        """
        Description:
        This Method computes the generalized multi-scale entropy moment.

        :param time: NumPy.ndarray - Contains the times, in [s], of each R-peak, in the 'rr_peaks' vector
        :param rr_peaks: NumPy.ndarray - Optional - if None and the EntropyManipulator was already initialized with an
                         RR-beats signal, wither in construction, or via using the 'insert_rr', and the rr_peaks input
                         is None, the EntropyManipulator will work on the existing signal. Default value is None.
        :param tau: int - Optional - This value determines the scale of the entropy moment to be computed. The default
                    value is 1. Please note that: max{tau} = length(rr_peaks) - 1, with order = 1.
        :param order: int - Optional - This value determines the order of the entropy 'derivative' to be computed.
                      Please note that: max{order} = length(rr_peaks) - 1, with tau = 1.
        :return: entropy_moment - NumPy.ndarray - Containing the computed entropy moment, according to the specified
                                  scale and order.
        """

        # Check for input errors
        if rr_peaks is None and self.rr_peaks is not None:
            rr_peaks = self.rr_peaks

        elif rr_peaks is None and self.rr_peaks is None:
            raise ValueError('No RR-Beat intervals signal was inserted to the Entropy Manipulator')

        # Calculate the  K-th order, n-th scale, entropy moment. At each iteration calculate the k-th order, n-th scale
        # moment, using the previously computed moment of order k - 1.
        entropy_moment = None
        length = rr_peaks.shape[0] - tau
        current_moment = np.zeros(length)
        for k in range(order):
            for i in range(length):
                if entropy_moment is None:
                    current_moment[i] = np.abs(rr_peaks[(i + tau)] - rr_peaks[i + tau - 1])

                else:
                    current_moment[i] = np.abs(entropy_moment[(i + tau)] - entropy_moment[i + tau - 1])

            time = time[tau:]
            entropy_moment = current_moment
            length = entropy_moment.shape[0] - tau
            current_moment = np.zeros(length)

            self.entropy = entropy_moment

        return time, entropy_moment

    def calc_mse(self, rr_peaks: np.ndarray = None, tau: int = 1):
        """
        Description:
        This method computes the multi-scale entropy measure, as defined by costa et al. at:
        Madalena Costa, Ary L. Goldberger, and C.-K. Peng -'Multiscale Entropy Analysis of Complex Physiologic Time
        Series' - Phys. Rev. Lett. 89, 068102, 19 July 2002.

        :param rr_peaks: NumPy.ndarray - Optional - if None and the EntropyManipulator was already initialized with an
                         RR-beats signal, wither in construction, or via using the 'insert_rr', and the rr_peaks input
                         is None, the EntropyManipulator will work on the existing signal. Default value is None.
        :param tau: int - Optional - This value determines the scale of the entropy moment to be computed. The default
                    value is 1. Please note that: max{tau} = length(rr_peaks) - 1.
        :return: mse - NumPy.ndarray - Containing the computed multi-scale entropy, according to the specified
                 scale.
        """

        # Check for input errors
        if rr_peaks is None and self.rr_peaks is not None:
            rr_peaks = self.rr_peaks

        elif rr_peaks is None and self.rr_peaks is None:
            raise ValueError('No RR-Beat intervals signal was inserted to the Entropy Manipulator')

        # Calculate k as a function of tau
        k = np.floor(rr_peaks.size / tau)

        # Calculate the multi-scale entropy
        mse = np.zeros_like([k, 1])
        for i in range(k):
            mse[k] = np.average(rr_peaks[(i * tau):((i + 1) * tau)])

        self.mse = mse

        return mse

    def __len__(self):
        """
        Description:
        Implementation of the inner __len__ method, which returns the length of the last signal insrted to the
        EntropyManipulator. To be used either by the user, or for using the EntropyManipulator as an Iterator.

        :return: self.rr_peaks.shape[0] - Int with the length of the signal on which the EntropyManipulator currently
                                          works on.
        """

        return self.rr_peaks.shape[0]

    def __getitem__(self, n: int):
        """
        Description:
        Implementation of the inner __getitem__ method. For the use of the EntropyManipulator as an Iterator object.

        :param n: int - Indicate the index of the element to be returned, from the currently inserted RR-beats signal.
        :return: self.rr_peaks[n] - NumPy.ndarray of shape (1,) - Containing the element at index n in the currently
                                    inserted RR-beats signal.
        """

        if n > self.__len__():
            raise StopIteration

        return self.rr_peaks[n]


def generate_entropy(save_dir: str, databases: list, file_name: str, start_ind: int = 120, entropies: ((int, int), ...)
                     = ((1, 2), (8, 2), (16, 2), (1, 3), (8, 3), (16, 3), (1, 4), (8, 4), (16, 4))):
    """
    Description:
    This functions takes in a list of paths to .pkl files containing wfdb databases, packed as DBHandler objects,
    and calculate the appropriate etnropy measures. It then saves them as an easy-to-use .pkl file for further
    computations

    :param save_dir: string - Specify the directory in which to save the calculated entropy measures.
    :param databases: list - Contains the paths to all of the .pkl files containing the relevant DBHandler objects.
    :param file_name: str -
    :param start_ind: int - Optional - How many initial R-Peaks to skip, in order to avoid operating on the initial,
                      noisy, parts of the recordings. Default value is 120.
    :param entropies: tuple - Optional - the tuple format should be ((tau_1, order_1), (tau_2, order_2), ...),
                      specifying the parameters for all of the entropy measures to be calculated. Default value is:
                      ((1, 2), (8, 2), (16, 2), (1, 3), (8, 3), (16, 3), (1, 4), (8, 4), (16, 4)).
    :return: None
    """

    # Setup
    print('Starting to load the database, please wait...\n')
    records_names = []
    entropy_calculator = EntropyManipulator()
    faulty_records = []
    entropies_list = []
    for i, tup in enumerate(entropies):
        print('\nCalculating Entropy Measure #' + str(i + 1) + ':')
        print('----------------------------------------------------\n')
        tau, order = tup
        entropy = []

        for k, db_path in enumerate(databases):
            begin_ind = db_path.find('Database\\') + len('Database\\')
            end_ind = db_path.find('.pkl')

            print('\nLoading Database #' + str(k + 1) + ' out of ' + str(len(databases)) + ' databases: ' +
                  db_path[begin_ind:end_ind])

            db = pickle.load(open(db_path, 'rb'))
            # database += db.db
            database = db.db
            r_names = [db_path[begin_ind:end_ind] + '_' + rec_name for rec_name in db.records_names]
            records_names += r_names

            del db
            # del db, databases
            fs = database[0].Fs

            # Calculated the entropies & appropriate time vectors
            db_entropy = []
            for j, record in enumerate(database):
                print('Calculating Entropy measure of record #' + str(j + 1) + ' out of ' + str(len(database)) +
                      ' records.')

                rr_peaks = record.annotations['R_peaks'][start_ind:] / fs

                # Check that the input signals are valid
                if not record.qrs_time[start_ind:].size == rr_peaks.size:
                    faulty_records.append(records_names[j])
                    continue

                time, ent = entropy_calculator.calc_multi_scale_entropy_moment(time=record.qrs_time[start_ind:],
                                                                               rr_peaks=rr_peaks, tau=tau, order=order)

                # Perform pre-proccessing prior to saving the final entropy
                inds = np.where(ent > (np.mean(ent) + 100 * np.std(ent)))

                if inds[0].size:
                    # Found some elements which needs to be corrected
                    inds = inds[0].tolist()

                    for ind in inds:
                        ent[ind] = np.mean(ent[(ind - 10):(ind + 10)])

                ent_tup = (time, ent)
                db_entropy.append(ent_tup)

            entropy += db_entropy

        entropies_list.append(entropy)

    # Save results
    print('\n Saving the Entropy Data...')
    objects = {'Entropy': entropies_list,
               'Records_Names': records_names,
               'Faulty_Records': faulty_records}

    file = open(os.path.join(save_dir, file_name + '.pkl'), 'wb')
    pickle.dump(obj=objects, file=file)
    file.close()


from DeterministicAnalytics.EntropyMethods.EntropyClustering.ClusterDB import create_features_matrices
from DeterministicAnalytics.EntropyMethods.Utilities.EKexp import EKexp

import numpy as np
import pickle
import os


class StatsHolder(object):
    def __init__(self):
        self.params = None
        self.f1_score = None
        self.precision = None
        self.recall = None
        self.true_pos = None
        self.false_pos = None
        self.true_neg = None
        self.false_neg = None


def experiment(db_paths: list, entropy_db_path: str, save_dir: str,
               alpha_vals: np.ndarray = np.linspace(1 / 250, 1 / 5, 10),
               winsize_vals: np.ndarray = np.linspace(20, 300, 6), win_len: int = 60,
               prediction_intervals: np.ndarray = np.linspace(20, 300, 6),
               n: np.ndarray = np.array((2, 10, 5)), edges: int = 180):
    """
    Description:

    :param db_paths:
    :param entropy_db_path:
    :param save_dir:
    :param alpha_vals:
    :param winsize_vals:
    :param win_len:
    :param prediction_intervals:
    :param n:
    :param edges:
    :return:
    """

    # Setup
    iterations = (alpha_vals.size * winsize_vals.size * prediction_intervals.size * n.size)
    ROC = np.zeros([iterations, 2])
    params = np.zeros([iterations, 4])
    statistics = []

    # Load Entropies
    entropy_db = pickle.load(open(entropy_db_path, 'rb'))
    entropies = entropy_db['Entropies']

    # Load RR-Peaks & Labels
    database = []
    labels = []
    names = []
    num_of_records = 0
    print('\nLoading the database...')
    for i, db_path in enumerate(db_paths):
        db = pickle.load(open(db_path, 'rb'))
        database.append(db.db)
        names.append(db.records_names)
        num_of_records += len(database[0])

        for record in database[i]:
            labels.append((record.annotations['AF_beats'] / record.Fs))

    del db

    # Calculate the references values
    features, times = create_features_matrices(entropies_list=entropies, remove_from_edges=edges)
    reference_stats = np.zeros((len(entropies), num_of_records))

    print('\nExtracting the features maps')
    for i, feature in enumerate(features):
        for j in range(num_of_records):
            reference_stats[i, j] = np.var(feature[j, :])

    # Create an EKexp class for each patient
    print('\nCreating the EKexp instances')
    ek_exps = []
    for k in range(len(features)):
        ek_exp = []

        for i in range(num_of_records):
            ek_exp.append(EKexp(af_times=labels[i], yyEnt_vec=features[k][:, i], yyTime=times[k][:, i],
                                window_length=win_len))

        ek_exps.append(ek_exp)

    # Run the main loop
    iteration = 0
    for i, alpha in enumerate(alpha_vals):
        for j, winsize in enumerate(winsize_vals):
            for k, interval in enumerate(prediction_intervals):
                for m, confidance in enumerate(n):
                    print('\nComputing Iteration #' + str(iteration + 1) + ' out of ' + str(iterations) + ' iterations')

                    true_pos = []
                    false_pos = []
                    true_neg = []
                    false_neg = []

                    # Analyze the database using the current parameters
                    for f in range(len(features)):
                        tp = 0
                        fp = 0
                        tn = 0
                        fn = 0
                        tp_fix_counter = 0

                        for r in range(num_of_records):
                            results = ek_exps[f][r].run_exp(alpha=alpha, prediction_interval=interval,
                                                            v0=reference_stats[f, r], running_avg_window_len=winsize,
                                                            n=confidance)

                            f_p, f_n, num_of_alarms, num_of_af_episodes = results

                            # if f_p != num_of_alarms - num_of_af_episodes - f_n:
                            #     f_p = num_of_alarms - num_of_af_episodes - f_n

                            if num_of_alarms == 0 and not num_of_af_episodes == 0:
                                if f_p == 0:
                                    tn += 1

                                else:
                                    tn += float(ek_exps[f][r].var_vec.size - f_p) / float(ek_exps[f][r].var_vec.size)

                                fn += f_n / num_of_af_episodes

                            elif num_of_af_episodes == 0 and not num_of_alarms == 0:
                                if f_p == 0:
                                    tn += 1

                                else:
                                    tn += float(ek_exps[f][r].var_vec.size - f_p) / float(ek_exps[f][r].var_vec.size)

                                fp += f_p / num_of_alarms

                            elif num_of_af_episodes == 0 and num_of_alarms == 0:
                                if f_p == 0:
                                    tn += 1

                                else:
                                    tn += float(ek_exps[f][r].var_vec.size - f_p) / float(ek_exps[f][r].var_vec.size)

                                tp_fix_counter += 1

                            else:
                                if f_p == 0:
                                    tn += 1

                                else:
                                    tn += float(ek_exps[f][r].var_vec.size - f_p) / float(ek_exps[f][r].var_vec.size)

                                tp += (num_of_alarms - f_p) / num_of_af_episodes
                                fp += f_p / num_of_alarms
                                fn += f_n / num_of_af_episodes

                        if tp_fix_counter == num_of_records:
                            tp = 0

                        else:
                            tp /= (num_of_records - tp_fix_counter)

                        fp /= num_of_records
                        tn /= num_of_records
                        fn /= num_of_records

                        true_pos.append(tp)
                        true_neg.append(tn)
                        false_pos.append(fp)
                        false_neg.append(fn)

                    # Save parameters
                    params[iteration, 0] = alpha
                    params[iteration, 1] = winsize
                    params[iteration, 2] = interval
                    params[iteration, 3] = confidance

                    # Save results
                    ROC[iteration, 0] = np.mean(true_pos)
                    ROC[iteration, 1] = np.mean(false_pos)

                    # Calculate statistics
                    precision = []
                    recall = []
                    f1_score = []
                    for p in range(len(features)):
                        if (true_pos[p] + false_pos[p]) == 0.0:
                            precision.append(0.0)

                        else:
                            precision.append(true_pos[p] / (true_pos[p] + false_pos[p]))

                        recall.append(true_pos[p] / (true_pos[p] + false_neg[p]))

                        if (precision[p] + recall[p]) == 0.0:
                            f1_score.append(0.0)

                        else:
                            f1_score.append(2 * (precision[p] * recall[p]) / (precision[p] + recall[p]))

                    # Save total statistics
                    stats_holder = StatsHolder()
                    stats_holder.params = [alpha, winsize, interval, confidance]
                    stats_holder.f1_score = f1_score
                    stats_holder.precision = precision
                    stats_holder.recall = recall
                    stats_holder.true_pos = true_pos
                    stats_holder.false_pos = false_pos
                    stats_holder.true_neg = true_neg
                    stats_holder.false_neg = false_neg

                    statistics.append(stats_holder)

                    # Save best parameters & score
                    if (i + j + k + m) == 0:
                        # First iteration
                        optimal_params = params[iteration, :]
                        optimal_f1_score = np.max(f1_score)
                        max_ent = np.argmax(f1_score)

                    else:
                        if np.max(f1_score) > optimal_f1_score:
                            # Found better parameters
                            optimal_params = params[iteration, :]
                            optimal_f1_score = np.max(f1_score)
                            max_ent = np.argmax(f1_score)

                    iteration += 1

    save_path = os.path.join(save_dir, 'grid_analysis_stats.pkl')
    file = open(save_path, 'wb')
    pickle.dump(obj=[optimal_params, statistics, max_ent], file=file)
    file.close()

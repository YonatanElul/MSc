from DeterministicAnalytics.EntropyMethods.Utilities.CalcEnt import generate_entropy

db_paths = [r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_1.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_2.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_3.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_4.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_5.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_6.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_7.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_8.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Normal_Sinus_Rythm_DB.pkl",
            r"B:\Studies\Master's\CardiacDiagnostics\Database\Atrial_Fibrillation_DB.pkl"
            ]

save_dir_paths = r"B:\Studies\Master's\CardiacDiagnostics\Database"

generate_entropy(save_dir=save_dir_paths, databases=db_paths, file_name='Ent_LT_NSR_AF_DBS',
                 entropies=((1, 2), (12, 2), (24, 2), (1, 4), (12, 4), (24, 4)))

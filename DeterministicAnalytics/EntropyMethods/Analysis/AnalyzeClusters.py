from DeterministicAnalytics.EntropyMethods.Optimizer.GridSearch import experiment


db = r"B:\Studies\Master's\CardiacDiagnostics\Database\Long-Term-Atrial-Fibrillation_DB_1.pkl"
ent_db = r"B:\Studies\Master's\CardiacDiagnostics\Database\EntropyDatabase.pkl"
save_dir = r"B:\Studies\Master's\CardiacDiagnostics\Database"

experiment(db_paths=[db], entropy_db_path=ent_db, save_dir=save_dir)

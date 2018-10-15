from ML.Utilities.CreateDBPickles import create_db_pickles


raw_data_dir = r"B:\Studies\Bachelor's\Semester 7\Project 1\Database\MIT-BIH Normal Sinus Rythm"
save_dir = r"B:\Studies\Master's\CardiacDiagnostics\Database"
db_name = r"Normal_Sinus_Rythm_DB"
db_type = 'Noise-Stress'

create_db_pickles(raw_data_dir=raw_data_dir, save_to_dir=save_dir, db_name=db_name, db_type=db_type)




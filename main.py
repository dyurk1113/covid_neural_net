import gen_data_files
import run_mon_neural_net
import process_results

if __name__ == '__main__':
    gen_data_files.rotate_policy_file()
    gen_data_files.build_unified_CA_df()
    gen_data_files.build_CA_training_df(max_date='2020-12-31')
    for trial in range(1):
        run_mon_neural_net.run_CA_UMNN_fit(val_method='date', trial_num=trial, output_folder='trial_data_files_tmp')
    gen_data_files.build_CA_training_df(max_date='2020-12-31', no_mon_col=True)
    for trial in range(1):
        run_mon_neural_net.run_CA_UMNN_fit(val_method='date', trial_num=trial, output_folder='trial_data_files_no_mon_tmp')
    process_results.do_full_processing()

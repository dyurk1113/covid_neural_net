from matplotlib import pyplot as plt

import gen_data_files
import run_neural_net
import process_results

if __name__ == '__main__':
    gen_data_files.rotate_policy_file()
    gen_data_files.build_unified_CA_df()
    gen_data_files.build_CA_training_df()
    run_neural_net.run_CA_UMNN_fit(val_method='date')

    # fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    # run_neural_net.run_CA_UMNN_fit(ax=axs[0], val_method='county')
    # run_neural_net.run_CA_UMNN_fit(ax=axs[1], val_method='date', ylab=False)
    # # fig.subplots_adjust(wspace=0, hspace=0)
    # fig.set_size_inches(9, 5)
    # fig.tight_layout()
    # fig.savefig('analysis_plots/fig_1_bw.tiff', bbox_inches='tight')

    # process_results.analyze_pol_eff_preds()

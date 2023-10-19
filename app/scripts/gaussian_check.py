import nestcheck.data_processing
import nestcheck.estimators as e
import nestcheck.diagnostics_tables
import pandas as pd

def main():
    base_dir = '/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/tests'  # directory containing run (PolyChord's 'base_dir' setting)
    file_roots = ['gaussian_2d_100nlive_r' + str(i) for i in range(1, 6)]

    run_list = nestcheck.data_processing.batch_process_data(
        file_roots, base_dir=base_dir, parallel=True,
        process_func=nestcheck.data_processing.process_polychord_run)

    print('The log evidence estimate using the first run is',
      e.logz(run_list[0]))
    print('The estimateed the mean of the first parameter is',
      e.param_mean(run_list[0], param_ind=0))

    estimator_list = [e.logz, e.param_mean, e.param_squared_mean, e.r_mean]
    # Use nestcheck's stored LaTeX format estimator names
    estimator_names = [e.get_latex_name(est) for est in estimator_list]
    vals_df = nestcheck.diagnostics_tables.estimator_values_df(
    run_list, estimator_list, estimator_names=estimator_names)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(vals_df)


if __name__ ==  '__main__':
    main()

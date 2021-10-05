import numpy as np
from DLP_Utils import flat_x_direction, weighed_dist, find_epsilon, flat_y_direction, cube_data_from_x_flat, \
    cube_data_from_y_flat
from scipy.spatial.distance import cdist


def run_ALP_model_train(data_train_start, data_train_end, label_train, num_iterations, alpha=0.5,
                        dist_method='euclidean', dim_ratio=[0.5, 0.5]):
    data_train_start_flat_x = flat_x_direction(data_train_start)
    square_dist_train_start_x = weighed_dist(data_train_start_flat_x, label_train, dist_method, 1)
    epsilon_train_start_x = find_epsilon(square_dist_train_start_x, alpha)

    data_train_flat_start_y = flat_y_direction(data_train_start)
    # pDist_train_flat_start_y = pdist(data_train_flat_start_y, 'euclidean')
    # square_dist_train_start_y = squareform(pDist_train_flat_start_y)
    square_dist_train_start_y = cdist(data_train_flat_start_y, data_train_flat_start_y, dist_method)
    epsilon_train_start_y = find_epsilon(square_dist_train_start_y, alpha)

    itter_epsilon_train_start_x = np.zeros((1, num_iterations + 1))
    itter_epsilon_train_start_x[0, 0] = epsilon_train_start_x
    itter_epsilon_train_start_y = np.zeros((1, num_iterations + 1))
    itter_epsilon_train_start_y[0, 0] = epsilon_train_start_y

    ALP_model = {'flat_x': data_train_start_flat_x,
                 'flat_y': data_train_flat_start_y,
                 'dist_method': dist_method,
                 'square_dist_x': square_dist_train_start_x,
                 'square_dist_y': square_dist_train_start_y,
                 'sigma_x': itter_epsilon_train_start_x,
                 'sigma_y': itter_epsilon_train_start_y,
                 'Kernel_x': np.zeros(
                     (np.shape(square_dist_train_start_x)[0], np.shape(square_dist_train_start_x)[1],
                      num_iterations)),
                 'Kernel_y': np.zeros(
                     (np.shape(square_dist_train_start_y)[0], np.shape(square_dist_train_start_y)[1],
                      num_iterations)),
                 'end_approx_x': [dict() for x in range(num_iterations)],
                 'end_approx_y': [dict() for x in range(num_iterations)],
                 'end_approx': [dict() for x in range(num_iterations)],
                 'end_multiscale': [dict() for x in range(num_iterations)],
                 'dist': [dict() for x in range(num_iterations)],
                 'error_itter': np.zeros((1, num_iterations)),
                 'root_error_itter': np.zeros((1, num_iterations))}
    # try:
    # Run the model on training cases
    if np.ndim(data_train_end) == 3:
        ALP_model = run_3D_ALP_model_train_itterations(ALP_model, data_train_end, num_iterations, dim_ratio)
    else:
        ALP_model = run_2D_ALP_model_train_itterations(ALP_model, data_train_end, num_iterations, dim_ratio)

    return ALP_model


def run_2D_ALP_model_train_itterations(ALP_model, data_end, num_itter=10, dim_ratio=[0.5, 0.5]):
    size_of_end = np.shape(data_end)[0] * np.shape(data_end)[1]
    data_end_flat_x = np.copy(data_end)
    data_end_flat_initial = np.reshape(data_end_flat_x, [size_of_end, 1], order='F')
    initial_data_end = np.copy(data_end)
    for itter in range(num_itter):

        sigma_x = ALP_model['sigma_x'][0, itter]
        data_end_flat_x = np.copy(data_end)
        [kernel_x, approx_flat_x] = run_ALP_1D(ALP_model['square_dist_x'], sigma_x, data_end_flat_x)
        ALP_model['Kernel_x'][:, :, itter] = kernel_x
        ALP_model['end_approx_x'][itter] = approx_flat_x

        data_end_flat_y = flat_y_direction(data_end)
        sigma_y = ALP_model['sigma_y'][0, itter]
        [kernel_y, approx_flat_y] = run_ALP_1D(ALP_model['square_dist_y'], sigma_y, data_end_flat_y)
        ALP_model['Kernel_y'][:, :, itter] = kernel_y
        ALP_model['end_approx_y'][itter] = approx_flat_y.T
        ALP_model['end_approx'][itter] = dim_ratio[0] * ALP_model['end_approx_x'][itter] + dim_ratio[1] * \
                                         ALP_model['end_approx_y'][itter]

        if itter == 0:
            ALP_model['end_multiscale'][itter] = ALP_model['end_approx'][itter]

        else:
            ALP_model['end_multiscale'][itter] = ALP_model['end_multiscale'][itter - 1] + ALP_model['end_approx'][itter]

        end_multiscale_flat_x = flat_x_direction(ALP_model['end_multiscale'][itter])
        end_multiscale_flat = np.reshape(end_multiscale_flat_x, [size_of_end, 1], order='F')

        err_vec_b = data_end_flat_initial - end_multiscale_flat
        err_vec_b = np.power(err_vec_b, 2)
        ALP_model['error_itter'][0, itter] = np.sum(err_vec_b) / size_of_end
        ALP_model['root_error_itter'][0, itter] = np.sqrt(ALP_model['error_itter'][0, itter])

        ALP_model['dist'][itter] = initial_data_end - ALP_model['end_multiscale'][itter]

        data_end = ALP_model['dist'][itter]
        ALP_model['sigma_x'][0, itter + 1] = ALP_model['sigma_x'][0, itter] / 2
        ALP_model['sigma_y'][0, itter + 1] = ALP_model['sigma_y'][0, itter] / 2

    return ALP_model


def run_3D_ALP_model_train_itterations(ALP_model, data_end, num_itter=10, dim_ratio=[0.5, 0.5]):
    size_of_end = np.shape(data_end)[0] * np.shape(data_end)[1] * np.shape(data_end)[2]
    data_end_flat_x = flat_x_direction(data_end)
    data_end_flat_initial = np.reshape(data_end_flat_x, [size_of_end, 1], order='F')
    initial_data_end = np.copy(data_end)
    for itter in range(num_itter):

        sigma_x = ALP_model['sigma_x'][0, itter]
        data_end_flat_x = flat_x_direction(data_end)
        [kernel_x, approx_flat_x] = run_ALP_1D(ALP_model['square_dist_x'], sigma_x, data_end_flat_x)
        ALP_model['Kernel_x'][:, :, itter] = kernel_x
        ALP_model['end_approx_x'][itter] = cube_data_from_x_flat(approx_flat_x, np.shape(data_end)[0],
                                                                 np.shape(data_end)[1], np.shape(data_end)[2])

        data_end_flat_y = flat_y_direction(data_end)
        sigma_y = ALP_model['sigma_y'][0, itter]
        [kernel_y, approx_flat_y] = run_ALP_1D(ALP_model['square_dist_y'], sigma_y, data_end_flat_y)
        ALP_model['Kernel_y'][:, :, itter] = kernel_y
        ALP_model['end_approx_y'][itter] = cube_data_from_y_flat(approx_flat_y, np.shape(data_end)[0],
                                                                 np.shape(data_end)[1], np.shape(data_end)[2])
        ALP_model['end_approx'][itter] = dim_ratio[0] * ALP_model['end_approx_x'][itter] + dim_ratio[1] * \
                                         ALP_model['end_approx_y'][itter]

        if itter == 0:
            ALP_model['end_multiscale'][itter] = ALP_model['end_approx'][itter]

        else:
            ALP_model['end_multiscale'][itter] = ALP_model['end_multiscale'][itter - 1] + ALP_model['end_approx'][itter]

        end_multiscale_flat_x = flat_x_direction(ALP_model['end_multiscale'][itter])
        end_multiscale_flat = np.reshape(end_multiscale_flat_x, [size_of_end, 1], order='F')

        err_vec_b = data_end_flat_initial - end_multiscale_flat
        err_vec_b = np.power(err_vec_b, 2)
        ALP_model['error_itter'][0, itter] = np.sum(err_vec_b) / size_of_end
        ALP_model['root_error_itter'][0, itter] = np.sqrt(ALP_model['error_itter'][0, itter])

        ALP_model['dist'][itter] = initial_data_end - ALP_model['end_multiscale'][itter]

        data_end = ALP_model['dist'][itter]
        ALP_model['sigma_x'][0, itter + 1] = ALP_model['sigma_x'][0, itter] / 2
        ALP_model['sigma_y'][0, itter + 1] = ALP_model['sigma_y'][0, itter] / 2

    return ALP_model


def run_ALP_1D(square_dist, sigma, data_end_flat):
    kernel = np.exp(-(np.power(square_dist, 2) / np.power(sigma, 2)))
    # zero diagonal
    for ind in range(len(kernel)):
        kernel[ind, ind] = 0
    # ave kernel
    for ind in range(len(kernel)):
        s = np.sum(kernel[ind, :])
        if s > 1e-6:
            kernel[ind, :] = kernel[ind, :] / s
        else:
            kernel[ind, :] = np.zeros((1, len(kernel)))

    approx_flat = np.zeros((np.shape(data_end_flat)[0], np.shape(data_end_flat)[1]))
    for row_ind in range(np.shape(data_end_flat)[0]):
        for col_ind in range(np.shape(data_end_flat)[0]):
            approx_flat[row_ind, :] = approx_flat[row_ind, :] + kernel[row_ind, col_ind] * data_end_flat[col_ind, :]

    return kernel, approx_flat


def run_ALP_model_test_iterations(ALP_model, all_data_test_start, all_data_test_end_true, data_train_end, min_ind):
    xs = np.shape(all_data_test_end_true)[0]
    ys = np.shape(all_data_test_end_true)[1]
    zs = np.shape(all_data_test_end_true)[2]
    end_test_multiscale_model = np.empty((xs, ys, zs))
    multi_scale_dict = {'s_model': [dict() for x in range(xs + 1)]}

    for test_s_ind in range(xs):
        single_test_start = all_data_test_start[test_s_ind, :, :]
        single_test_end_true = all_data_test_end_true[test_s_ind, :, :]
        end_test_model = {'multiscale': [dict() for x in range(min_ind + 1)],
                          'approx': [dict() for x in range(min_ind + 1)],
                          'mse_error': np.zeros((1, min_ind + 1)),
                          'root_mse_error': np.zeros((1, min_ind + 1))}  # REMOVE THE (-1) WHEN ITS NOT THE 0 INDEX!!!

        data_test_end = np.copy(data_train_end)
        # k_i = 0
        # if np.ndim(single_test_start) == 3:
        #    x_flat_s = np.shape(single_test_start)[0]
        #    y_flat_s = np.shape(single_test_start)[1] * np.shape(single_test_start)[2]
        #    x_flat_e = np.shape(single_test_end_true)[0]
        #    y_flat_e = np.shape(single_test_end_true)[1] * np.shape(single_test_end_true)[2]
        #    data_test_end_true_flat = np.reshape(single_test_end_true, [x_flat_e, y_flat_e])
        # elif np.ndim(single_test_start) == 2:
        x_flat_s = 1
        y_flat_s = np.shape(single_test_start)[0] * np.shape(single_test_start)[1]
        x_flat_e = 1
        y_flat_e = np.shape(single_test_end_true)[0] * np.shape(single_test_end_true)[1]
        data_test_end_true_flat = np.reshape(single_test_end_true, [x_flat_e, y_flat_e], order='F')

        for k_i in range(min_ind + 1):

            date_test_start_flat_x = np.reshape(single_test_start, [x_flat_s, y_flat_s], order='F')

            p_dist = cdist(ALP_model['flat_x'], date_test_start_flat_x, ALP_model['dist_method'])

            kernel_x_test = np.exp(-(np.power(p_dist, 2) / np.power(ALP_model['sigma_x'][0, k_i], 2)))
            for ind in range(np.shape(kernel_x_test)[1]):
                s = np.sum(kernel_x_test[:, ind])
                if s > 1e-6:
                    kernel_x_test[:, ind] = kernel_x_test[:, ind] / s
                else:
                    kernel_x_test[:, ind] = np.zeros((1, len(kernel_x_test)))

            data_test_end_flat_x = flat_x_direction(data_test_end)
            end_flat_for_kernel_X = np.zeros([x_flat_s,
                                              np.shape(data_test_end)[1] * np.shape(data_test_end)[2]])

            for subject_ind in range(np.shape(kernel_x_test)[1]):
                for row_ind in range(np.shape(kernel_x_test)[0]):
                    end_flat_for_kernel_X = end_flat_for_kernel_X + \
                                            kernel_x_test[row_ind, subject_ind] * data_test_end_flat_x[row_ind, :]

            end_test_model['approx'][k_i] = cube_data_from_x_flat(end_flat_for_kernel_X, x_flat_s,
                                                                  np.shape(data_train_end)[1],
                                                                  np.shape(data_train_end)[2])

            if k_i == 0:
                end_test_model['multiscale'][k_i] = end_test_model['approx'][k_i]
            else:
                end_test_model['multiscale'][k_i] = end_test_model['multiscale'][k_i - 1] + end_test_model['approx'][
                    k_i]

            data_test_end = np.copy(ALP_model['dist'][k_i])

            end_test_multiscale_flat = np.reshape(end_test_model['multiscale'][k_i], [x_flat_e, y_flat_e], order='F')
            error_vec_test = data_test_end_true_flat - end_test_multiscale_flat
            error_vec_test_2 = np.power(error_vec_test, 2)
            end_test_model['mse_error'][0, k_i] = np.sum(error_vec_test_2) / y_flat_e
            end_test_model['root_mse_error'][0, k_i] = np.sqrt(end_test_model['mse_error'][0, k_i])

        multi_scale_dict['s_model'][test_s_ind] = end_test_model
        end_test_multiscale_model[test_s_ind, :, :] = end_test_model['multiscale'][min_ind]

    return end_test_multiscale_model, multi_scale_dict

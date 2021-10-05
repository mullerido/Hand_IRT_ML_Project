import numpy as np
from scipy.spatial.distance import cdist


def running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1

        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def flat_x_direction(data):

    if np.ndim(data) == 2:
        flat_data = data
    else:
        shape = np.shape(data)

        flat_data = np.empty((shape[0], shape[1] * shape[2]))
        for t_i in range(shape[2]):
            current_time = data[:, :, t_i]
            flat_data[:, t_i * shape[1]:t_i * shape[1] + shape[1]] = current_time
    return flat_data


def flat_y_direction(data):

    if np.ndim(data) == 2:
        flat_data_out = data.T
    else:
        shape = np.shape(data)
        flat_data = np.empty((0, shape[1]), float)  # np.empty((shape[0]*shape[2], shape[1]))
        for t_i in range(shape[2]):
            current_time = data[:, :, t_i]
            flat_data = np.append(flat_data, current_time, axis=0)

        flat_data_out = flat_data.T
    return flat_data_out


def find_epsilon(dist_data, alpha=0.5):
    """
    % This is the huristics used to find epsilon
    (see this paper: Heterogeneous Datasets Representation and Learning using Diffusion Maps and Laplacian Pyramids.N Rabin, RR Coifman, SDM, 189 - 199)

    % For input send:
        dist = squareform(pdist(YourInputData));
    % For other
        metrics: dist = squareform(pdist(YourInputData, 'cosine'));
    """
    shape = np.shape(dist_data)
    I_mat = np.identity(shape[0])
    max_val = np.max(dist_data)
    mins = np.min(dist_data + I_mat * max_val, axis=1)

    eps = np.max(mins) * alpha

    return eps


def cube_data_from_x_flat(flat_x, x, y, z):
    cube_data = np.zeros((x, y, z))
    j = 1
    for t_ind in range(z):
        t_values = flat_x[:, (j - 1) * y: j * y]
        cube_data[:, :, t_ind] = t_values
        j += 1

    return cube_data


def cube_data_from_y_flat(flat_y, x, y, z):
    cube_data = np.zeros((x, y, z))
    flat_y = flat_y.T
    i = 1
    for t_ind in range(z):
        t_values = flat_y[(i - 1) * x:i * x, :]
        cube_data[:, :, t_ind] = t_values
        i += 1
    return cube_data


def convergence_ROIs(data_cube):
    if np.ndim(data_cube) == 3:
        data_2D = np.zeros([np.shape(data_cube)[0], np.shape(data_cube)[2]])
        for s_ind in range(np.shape(data_cube)[0]):
            for t_ind in range(np.shape(data_cube)[2]):
                data_2D[s_ind, t_ind] = np.mean(data_cube[s_ind, :, t_ind])

    elif np.ndim(data_cube) == 2:
        data_2D = data_cube.mean(axis=0)
        data_2D = data_2D.reshape(1, -1)

    return data_2D


def weighed_dist(data, labels, dist_method='euclidean', w=1):
    """
        Calculate the distance matrix of data and apply weight based on the label
    """
    # dist_list = pdist(data, 'euclidean')
    # dist_sqare = squareform(dist_list)
    dist_sqare = cdist(data, data, dist_method)

    unique_labels = np.unique(labels)
    for c_l in unique_labels:
        same_label_ids = np.where(labels == c_l)
        for r in same_label_ids[0]:
            for c in same_label_ids[0]:
                dist_sqare[r, c] = dist_sqare[r, c] * w

    return dist_sqare


def get_ave_of_k_nearest_data(data_train, data_test, n):
    closest_data = np.empty((np.shape(data_test)[0], np.shape(data_test)[2]))
    # Average ROIs per patient- from cube 2 2D
    data_train_2D = convergence_ROIs(data_train)
    data_test_2D = convergence_ROIs(data_test)
    # if np.ndim(data_test) == 3:
    #    end_model_2D = convergence_ROIs(end_model)
    # elif np.ndim(data_test) == 2:
    #    end_model_2D = convergence_ROIs(end_model.reshape((5, 13), order='F'))
    x = np.linspace(0, np.shape(data_test_2D)[1] - 1, np.shape(data_test_2D)[1])

    for test_s_ind in range(np.shape(data_test_2D)[0]):
        dist_to_test = cdist(data_test_2D[test_s_ind, 0:n].reshape(1, -1), data_train_2D[:, 0:n], 'euclidean')

        min_arr = np.argpartition(dist_to_test, 3, axis=1)[0][:3]

        closest_data[test_s_ind, :] = data_train_2D[min_arr, :].mean(axis=0)
        '''
        fig, ax = plt.subplots()
        ax.plot(x, data_train_2D[min_arr[0], :], linestyle='-', linewidth=1, color='C1')
        ax.plot(x, data_train_2D[min_arr[1], :], linestyle='-', linewidth=1, color='C1')
        ax.plot(x, data_train_2D[min_arr[2], :], linestyle='-', linewidth=1, color='C1')
        ax.plot(x, closest_data[test_s_ind, :], linestyle='-', linewidth=1, color='C2')
        '''
    return closest_data
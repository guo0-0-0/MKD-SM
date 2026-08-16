################################## 泊松圆盘采样 37*37网格 ########################################################
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os

def poisson_disk_sampling(width, height, min_dist, sample_count, initial_point, k=50):
    cell_size = min_dist / np.sqrt(2)

    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [[-1 for _ in range(grid_height)] for _ in range(grid_width)]

    samples = []
    active_list = []

    initial_x, initial_y = initial_point
    samples.append(initial_point)
    active_list.append(initial_point)
    grid_x = int(initial_x / cell_size)
    grid_y = int(initial_y / cell_size)
    grid[grid_x][grid_y] = 0

    while active_list and len(samples) < sample_count:
        idx = np.random.randint(0, len(active_list))
        center = active_list[idx]
        found = False

        for _ in range(k):
            radius = np.random.uniform(min_dist, min_dist * 1.5)
            angle = np.random.uniform(0, 2 * np.pi)
            new_x = center[0] + radius * np.cos(angle)
            new_y = center[1] + radius * np.sin(angle)
            new_point = (new_x, new_y)

            if 0 <= new_x < width and 0 <= new_y < height:
                grid_x = int(new_x / cell_size)
                grid_y = int(new_y / cell_size)

                too_close = False
                for i in range(max(0, grid_x - 2), min(grid_width, grid_x + 3)):
                    for j in range(max(0, grid_y - 2), min(grid_height, grid_y + 3)):
                        neighbor_idx = grid[i][j]
                        if neighbor_idx != -1:
                            neighbor = samples[neighbor_idx]
                            distance = np.hypot(new_x - neighbor[0], new_y - neighbor[1])
                            if distance < min_dist:
                                too_close = True
                                break
                    if too_close:
                        break

                if not too_close:
                    samples.append(new_point)
                    active_list.append(new_point)
                    grid[grid_x][grid_y] = len(samples) - 1
                    found = True
                    break

        if not found:
            active_list.pop(idx)

    return samples[:sample_count]

def map_samples_to_grid(samples, width, height):
    grid_indices_2d = []
    grid_indices_1d = []
    for point in samples:
        x_idx = int(point[0])
        y_idx = int(point[1])
        x_idx = min(max(x_idx, 0), width - 1)
        y_idx = min(max(y_idx, 0), height - 1)
        grid_indices_2d.append((x_idx, y_idx))
        grid_indices_1d.append(y_idx * width + x_idx)

    sorted_indices = sorted(range(len(grid_indices_2d)), key=lambda i: (grid_indices_2d[i][1], grid_indices_2d[i][0]))
    grid_indices_2d = [grid_indices_2d[i] for i in sorted_indices]
    grid_indices_1d = [grid_indices_1d[i] for i in sorted_indices]

    return grid_indices_2d, grid_indices_1d

grid_width = 37
grid_height = 37
down = 8

if down == 2:
    sample_count = 343
elif down == 4:
    sample_count = 86
else:
    sample_count = 22

approx_area_per_point = (grid_width * grid_height) / sample_count
min_dist = np.sqrt(approx_area_per_point / np.pi) * 1.5  # Increase min_dist for more uniform sampling
sampling_index_list_1d = []
sampling_index_list_2d = []

initial_point = (18, 18)
for i in range(37):

    # initial_point = (np.random.uniform(0, grid_width), np.random.uniform(0, grid_height))

    samples = poisson_disk_sampling(grid_width, grid_height, min_dist, sample_count, initial_point)

    grid_indices_2d, grid_indices_1d = map_samples_to_grid(samples, grid_width, grid_height)

    sampling_index_list_1d.append(grid_indices_1d)
    sampling_index_list_2d.append(grid_indices_2d)

    print(grid_indices_1d)
    print('len grid_indices_1d: ', len(grid_indices_1d))

sampling_index_all_slice = np.stack(sampling_index_list_1d, axis=0)
print('sampling_index_all_slice shape', sampling_index_all_slice.shape)

grid_indices_all_slice_1d = []
for i in range(sampling_index_all_slice.shape[0]):
    for j in range(sampling_index_all_slice.shape[1]):
        grid_indices_all_slice_1d.append(sampling_index_all_slice[i][j] + i * 1369)

print(grid_indices_all_slice_1d)
print('len grid_indices_all_slice_1d: ', len(grid_indices_all_slice_1d))

sampled_path = r'/media/poisson_disk_sampling/poisson_disk_sampled_down' + str(down) + '_37_all_slice_1d.pkl'
pickle.dump(grid_indices_all_slice_1d, open(sampled_path, 'wb'))



















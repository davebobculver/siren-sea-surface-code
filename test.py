import os
import shutil
import numpy as np

os.makedirs('temp_dir', exist_ok=True)

save_dict = {}

for i in range(100):
    data = np.random.rand(10000, 1000)

    if i > 40:
        filename = f'temp_dir/data_{i}.npy'
        np.save(filename, data)

        mean = data.mean()
        save_dict[filename] = mean

        # Keep only the two lowest means
        if len(save_dict) > 5:
            # find file with highest mean
            worst_file = max(save_dict, key=save_dict.get)

            # delete from disk
            os.remove(worst_file)

            # delete from dictionary
            del save_dict[worst_file]

# Now we have only the 2 lowest-mean matrices left
# Find the absolute lowest
best_file = min(save_dict, key=save_dict.get)

best_matrix = np.load(best_file)
np.save('best_matrix.npy', best_matrix)

# Clean up directory
shutil.rmtree('temp_dir')

# best_matrix is still in memory and safe
print("Lowest mean:", save_dict[best_file])

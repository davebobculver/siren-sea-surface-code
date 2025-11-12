import h5py
import pickle

# class StructObject:
#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

# Load the .mat file using h5py
mat_data = h5py.File(r'Data\20241218_flight1_hover1.mat', 'r')

# Access the cell array
cell_array = mat_data['FRAME']

print("Type of cell_array:", type(cell_array))

# Function to dereference HDF5 references
def dereference(h5py_obj):
    if isinstance(h5py_obj, h5py.h5r.Reference):
        referenced_obj = mat_data[h5py_obj]
        return dereference(referenced_obj)
    elif isinstance(h5py_obj, h5py.Dataset):
        return h5py_obj[()]
    elif isinstance(h5py_obj, h5py.Group):
        return {name: dereference(item) for name, item in h5py_obj.items()}
    else:
        return h5py_obj

# Function to convert HDF5 structured arrays to StructObject instances
def h5py_struct_array_to_objects(struct_array):

    obj = { 'X': dereference(struct_array)['XYZ'][0],
           'Y': dereference(struct_array)['XYZ'][1],
           'Z': dereference(struct_array)['XYZ'][2],
           'Amplitude': dereference(struct_array)['Amplitude'],
           'time_dnum': dereference(struct_array)['time_dnum'],
           'time_sec': dereference(struct_array)['time_sec'],
           'Beamnum': dereference(struct_array)['Beamnum'],
           'Distance': dereference(struct_array)['Distance']
    }
    
    return obj

# print("Type of cell_array[0]:", type(dereference(cell_array[0])))
# print("Type of cell_array[0][0]:", type(dereference(cell_array[0][0])))
# print("keys of cell_array[0][0]:", list(dereference(cell_array[0][0]).keys()))


# print("Cell array XYZ:", (dereference(cell_array[0][0]))['XYZ'][1]  )

# Convert the cell array to a list of StructObject instances
list_of_objects = []
test = 0
while test< 9600:
    cell = cell_array[test]
    struct = dereference(cell[0])
    obj = h5py_struct_array_to_objects(struct)
    list_of_objects.append(obj)
    
    if test == 0:
        print("First object:", list_of_objects[0])
        print("Type of first object:", type(list_of_objects[0]))

    if test% 100 == 0:
        print(f"Processed {test} objects")
    test += 1
    

# Save the list of objects as a pickle file
output_file = r'Data\Data_20241218.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(list_of_objects, f)

# Close the .mat file
mat_data.close()

print(f"Data successfully converted and saved to {output_file}")
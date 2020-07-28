skip = 0
dataset_path = 'C:\\Users\\DELL\\Desktop\\python_file\\'

face_data = [] 
labels = []

class_id = 0 # Labels for specific face
names = {} #map with class_id


# Data Preparation
for file in os.listdir(dataset_path):
	if file.endswith('.npy'):
		#Create a mapping btw class_id and name
		names[class_id] = file[:-4]
		print("Loaded "+file)
		data_item = np.load(dataset_path+file)
		face_data.append(data_item)

		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

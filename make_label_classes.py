import numpy as np

labels = np.array(['benign', 'suspicious', 'precancerous', 'cancerous'])
np.save("model/label_classes.npy", labels)
print("label_classes.npy created successfully:", labels)

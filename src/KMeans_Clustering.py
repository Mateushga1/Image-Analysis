import numpy as np
import cv2

from sklearn.cluster import KMeans

#img_bgr=cv2.imread('Images_Classif/Essex_Faces/94/ajsega.19.jpg')
img_bgr=cv2.imread('Parrots.jpg')
(h_img,w_img,c) = img_bgr.shape
print("Dimension de l'image :",h_img,"lignes x",w_img,"colonnes x",c,"canaux")
print("Type de l'image :",img_bgr.dtype)

# Création des clusters (entraînement) sur une image
Nb_classes = 20
img_samples = np.reshape(img_bgr,(-1,3))
kmeans = KMeans(n_clusters=Nb_classes, random_state=0).fit(img_samples)
# Affichage des centres de cluster 
print("Centres des clusters : ",kmeans.cluster_centers_)
# Affichage des labels dans l'image d'entraînement
img_labels = np.reshape(kmeans.labels_,(h_img,w_img))
print("Type de l'image de label :",img_labels.dtype)
# Normalisation pour affichage
img_labels_display = (img_labels*255)/(Nb_classes - 1)
img_labels_display = img_labels_display.astype(np.uint8)
cv2.imshow("Clusters dans l'image (train)",img_labels_display)
cv2.waitKey(0)

# Charger la nouvelle image pour les tests
#new_img_bgr = cv2.imread('Images_Classif/Essex_Faces/94/ccjame.19.jpg')
new_img_bgr = cv2.imread('Parrots.jpg')

# Redimensionner aux mêmes dimensions que l'image d'entraînement
new_img_bgr = cv2.resize(new_img_bgr, (w_img, h_img))

# Convertir la nouvelle image dans le format approprié
new_img_samples = np.reshape(new_img_bgr, (-1, 3))

# Appliquer le modèle K-means entraîné sur la nouvelle image
new_img_labels = kmeans.predict(new_img_samples)

# Redimensionner les étiquettes aux dimensions de l'image
new_img_labels_display = np.reshape(new_img_labels, (h_img, w_img))

# Normalisation pour affichage
new_img_labels_display = (new_img_labels_display * 255) / (Nb_classes - 1)
new_img_labels_display = new_img_labels_display.astype(np.uint8)
cv2.imshow("Clusters dans l'image (test)", new_img_labels_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

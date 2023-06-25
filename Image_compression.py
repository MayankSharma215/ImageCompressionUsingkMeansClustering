import numpy as np
import matplotlib.pyplot as plt


def find_closest_centroids(X, centroids):
    
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)
    dist = np.zeros(K, dtype=float)

    for i in range(len(X)):
        for j in range(K):
            dist[j]=np.linalg.norm(X[i]-centroids[j])
        idx[i]=np.argmin(dist)
    
    return idx


def compute_centroids(X, idx, K):
    
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    
    for i in range(K):
        centroids[i]=np.mean([x for j,x in enumerate(X) if idx[j]==i],axis=0) 
    
    return centroids



def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    for i in range(max_iters):
        
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        idx = find_closest_centroids(X, centroids)

        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx




def kMeans_init_centroids(X, K):
    
    randidx = np.random.permutation(X.shape[0])
    
    centroids = X[randidx[:K]]
    
    return centroids



original_img = plt.imread('Image Compression/bird_small.png')
plt.imshow(original_img)

print("Shape of original_img is:", original_img.shape)        

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))



K = 16
max_iters = 10

initial_centroids = kMeans_init_centroids(X_img, K)

centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)



print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])



# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :] 

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 



# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
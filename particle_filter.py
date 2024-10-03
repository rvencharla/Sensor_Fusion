import numpy as np

class ParticleFilter:
    '''
    Particle Filter Class
    '''
    def __init__(self):
        self.num_particles = 1000
        self.dim_state = 2 # process model dimension
        self.dt = 1 # time increment
        self.q= 0.5 # process noise variable Q
        self.particles = np.random.rand(self.num_particles, self.dim_state)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def A(self):
        # system matrix
        dt = self.dt
        return np.matrix([[1,0,1,1], 
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])

    def Q(self):
        # Q noise follwing Gaussian distribution
        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        return np.matrix([[q1, 0, q2, 0],
                        [0, q1, 0, q2],
                        [q2, 0, q3, 0],
                        [0, q2, 0,  q3]])
        
    def C(self):
        # measurement matrix C, true state space
        return np.matrix([[1,0, 0, 0],
                         [0,1, 0, 0]])
    
    def predict(self):
        self.particles = np.linalg.inv(np.eye(2)-0.01*self.A()) + self.Q()

    def update(self, z):
        # z is data
        weights = np.zeros(self.num_particles)
        for i in range(len(self.particles)):
            likelihood = self.likelihood(self.particles[i], z, self.Q())
            weights[i] = likelihood

        # Normalize weights
        weights /= sum(weights)

        return weights

    def resample(self, weights_array):
        n = len(weights_array)
        weight_sums = []
        weight_sums.append(weights_array[0])
        for i in range(n-1):
            weight_sums.append(weight_sums[i] + weights_array[i+1])
        
        start = np.random.uniform(low=0.0, high=1/(n))
        resampled_indicies = []
        for j in range(n):
            curr = start + (1/n) * j
            size = 0
            while curr > weight_sums[size]:
                size += 1
                resampled_indicies.append(size)
        return resampled_indicies
    
    def run_filter_pf(self, gps_data, vio_data):
        for t in range(len(gps_data)):
            self.particles = self.predict()

            weights = self.update(gps_data[t])

            # Resample
            indexes = self.resample(weights)
            self.particles[:] = self.particles[indexes]
            self.weights.fill(1.0 / len(self.weights))

    def likelihood(self, particle, data, noise):
        """
        Compute the likelihood of the data given a particle.
        The likelihood is assumed to follow a Gaussian distribution.
        """
        mean = particle
        cov = noise
        diff = data - mean

        # Compute the Gaussian probability density function
        likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
        likelihood /= np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))

        return likelihood
    
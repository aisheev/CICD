import numpy as np
def run_standard_pso(obj_func, n_particles=30, n_dimensions=2, max_iter=100, bounds=(-10, 10), w_max=0.9, w_min=0.4):
    # Initialize particles and velocities
    particles = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dimensions))
    velocity = np.zeros((n_particles, n_dimensions))
    
    # Personal best positions and values
    pbest = np.copy(particles)
    pbest_values = np.array([obj_func(x) for x in particles])
    
    # Global best position and value
    gbest_value = np.min(pbest_values)
    gbest = particles[np.argmin(pbest_values)]
    
    # Velocity clamping
    v_max = (bounds[1] - bounds[0]) * 0.2  # Clamping factor

    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter  # Linearly decreasing inertia weight

        for i in range(n_particles):
            r1, r2 = np.random.rand(2)  # Random coefficients

            # Update velocities
            velocity[i] = w * velocity[i] + \
                          1.49445 * r1 * (pbest[i] - particles[i]) + \
                          1.49445 * r2 * (gbest - particles[i])
            
            # Apply velocity clamping
            velocity[i] = np.clip(velocity[i], -v_max, v_max)
            
            # Update positions
            particles[i] += velocity[i]
            
            # Boundary condition handling
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            # Update personal bests
            current_value = obj_func(particles[i])
            if current_value < pbest_values[i]:
                pbest[i] = particles[i]
                pbest_values[i] = current_value
                
                # Update global best
                if current_value < gbest_value:
                    gbest = particles[i]
                    gbest_value = current_value

    return gbest, gbest_value



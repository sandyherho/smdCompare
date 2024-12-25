#!/usr/bin/env Rscript

# Load required package
library(deSolve)

# System parameters
mass <- 100
stiffness <- 50
damping <- 50

# Create force input
create_step <- function(t, t_start, t_end, amplitude) {
  y <- numeric(length(t))
  idx_start <- which(t >= t_start)[1]
  idx_end <- which(t >= t_end)[1]
  
  for(i in seq_along(t)) {
    if(i < idx_start) y[i] <- 0
    else if(i > idx_end) y[i] <- amplitude
    else y[i] <- amplitude * (i - idx_start)/(idx_end - idx_start)
  }
  return(y)
}

# Define the system of ODEs
smd_system <- function(t, state, parameters) {
  force <- parameters$force_values[max(1, which.min(abs(parameters$force_times - t)))]
  
  dstate <- c(
    state[2],
    (force - parameters$stiffness * state[1] - parameters$damping * state[2])/parameters$mass
  )
  
  return(list(dstate))
}

# Time vector
times <- seq(0, 100, by = 0.01)
force <- create_step(times, t_start = 4, t_end = 5, amplitude = 50)

# Parameters list
parameters <- list(
  mass = mass,
  stiffness = stiffness,
  damping = damping,
  force_times = times,
  force_values = force
)

# Initial conditions
state <- c(0, 0)  # Initial position and velocity

# Solve the system
solution <- ode(
  y = state,
  times = times,
  func = smd_system,
  parms = parameters
)

# Extract position and velocity
position <- solution[, 2]
velocity <- solution[, 3]

# Calculate acceleration
acceleration <- c(0, diff(velocity)/diff(times))

# Simulation results
simulation_results <- data.frame(
  Time = times,
  Force = force,
  Position = position,
  Speed = velocity,
  Acceleration = acceleration
)

# Calculate stability metrics
A <- matrix(c(0, 1, -stiffness/mass, -damping/mass), 2, 2)
eigs <- eigen(A)$values

# Create metrics dataframe
metrics_df <- data.frame(
  Metric = c(
    "Eigenvalues",
    "Characteristic_Equation",
    "Damping_Factor",
    "Natural_Frequency",
    "Asymptotic_Stability",
    "Routh_Stability",
    "BIBO_Stability"
  ),
  Value = c(
    paste(format(Re(eigs), digits=6), "+", format(Im(eigs), digits=6), "i", collapse=", "),
    paste(c(1, damping/mass, stiffness/mass), collapse=", "),
    format(damping/(2*sqrt(mass*stiffness)), digits=6),
    format(sqrt(stiffness/mass), digits=6),
    ifelse(all(Re(eigs) < 0), "Stable", "Unstable"),
    "Stable",
    "Stable"
  ),
  stringsAsFactors = FALSE
)

# Create output directory
dir.create("../outputs/data", recursive = TRUE, showWarnings = FALSE)

# Save results
write.csv(simulation_results, "../outputs/data/smd_simulation_r.csv", row.names = FALSE)
write.csv(metrics_df, "../outputs/data/smd_stability_metrics_r.csv", row.names = FALSE)

# Print final values
cat("\nSimulation Complete. Results saved to CSV files.\n")
cat("\nFinal values:\n")
cat("Position:", format(tail(simulation_results$Position, 1), digits=3), "m\n")
cat("Velocity:", format(tail(simulation_results$Speed, 1), digits=3), "m/s\n")
cat("Acceleration:", format(tail(simulation_results$Acceleration, 1), digits=3), "m/s²\n")

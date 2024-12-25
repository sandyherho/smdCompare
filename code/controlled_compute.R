# Spring Mass Damper System with PID Control in R

library(deSolve)
library(dplyr)

create_step_input <- function(time, y_start, t_start, y_end, t_end) {
  y <- rep(0, length(time))
  
  idx_start <- which(time >= t_start)[1]
  idx_end <- which(time >= t_end)[1]
  
  slope <- (y_end - y_start) / (t_end - t_start)
  
  y[idx_start:idx_end] <- y_start + slope * (time[idx_start:idx_end] - t_start)
  y[idx_end:length(time)] <- y_end
  
  return(y)
}

smd_system <- function(t, state, params) {
  with(as.list(c(state, params)), {
    position <- state[1]
    velocity <- state[2]

    acceleration <- (control_input - k * position - d * velocity) / m

    return(list(c(velocity, acceleration)))
  })
}

pid_controller <- function(setpoint, measurement, time, state) {
  error <- setpoint - measurement

  dt <- ifelse(is.null(state$last_time), 0, time - state$last_time)
  
  if (dt > 0) {
    state$integral <- state$integral + error * dt
    derivative <- (error - state$last_error) / dt
  } else {
    derivative <- 0
  }
  
  output <- state$kp * error + state$ki * state$integral + state$kd * derivative

  state$last_error <- error
  state$last_time <- time

  return(list(output = output, state = state))
}

analyze_closed_loop_stability <- function(params) {
  m <- params$m
  k <- params$k
  d <- params$d
  kp <- params$kp
  ki <- params$ki
  kd <- params$kd

  char_poly <- c(1, (d + kd) / m, (k + kp) / m, ki / m)

  eigenvals <- eigen(matrix(c(0, 1, 0,
                              -k / m, -d / m, 0,
                              -kp / m, -kd / m, -ki / m),
                            nrow = 3, byrow = TRUE))$values

  is_stable <- all(Re(eigenvals) < 0)

  routh_array <- matrix(0, nrow = 4, ncol = 2)
  routh_array[1, ] <- c(char_poly[1], char_poly[3])
  routh_array[2, ] <- c(char_poly[2], char_poly[4])
  routh_array[3, 1] <- (routh_array[2, 1] * routh_array[1, 2] - routh_array[1, 1] * routh_array[2, 2]) / routh_array[2, 1]
  routh_array[4, 1] <- char_poly[4]
  routh_stable <- all(routh_array[, 1] > 0)

  bibo_stable <- all(char_poly > 0)

  metrics <- list(
    "Eigenvalues" = eigenvals,
    "Characteristic Equation" = char_poly,
    "Natural Frequency" = sqrt((k + kp) / m),
    "Damping Ratio" = (d + kd) / (2 * sqrt(m * (k + kp))),
    "Asymptotic Stability" = ifelse(is_stable, "Stable", "Unstable"),
    "Routh Stability" = ifelse(routh_stable, "Stable", "Unstable"),
    "BIBO Stability" = ifelse(bibo_stable, "Stable", "Unstable")
  )

  return(metrics)
}

simulate_with_control <- function(params, sim_params) {
  # Initialize PID state
  pid_state <- list(kp = params$kp, ki = params$ki, kd = params$kd, 
                    last_error = 0, integral = 0, last_time = NULL)

  time <- seq(sim_params$start_time, sim_params$end_time, sim_params$step_size)
  desired_positions <- create_step_input(time, sim_params$input_start_val, 
                                         sim_params$input_start, 
                                         sim_params$input_end_val, 
                                         sim_params$input_end)

  initial_state <- c(position = 0, velocity = 0)
  control_inputs <- numeric(length(time))

  state <- initial_state

  results <- data.frame(Time = time, Position = NA, Speed = NA, Control_Input = NA, Desired_Position = desired_positions)

  for (i in seq_along(time)) {
    t <- time[i]
    desired_position <- desired_positions[i]

    control_result <- pid_controller(desired_position, state["position"], t, pid_state)
    pid_state <- control_result$state
    control_inputs[i] <- control_result$output

    params$control_input <- control_result$output

    state <- ode(y = state, times = c(t, t + sim_params$step_size), 
                 func = smd_system, parms = params, method = "rk4")[2, 2:3]

    results[i, ] <- c(t, state["position"], state["velocity"], control_inputs[i], desired_position)

    # Debugging: Print the current state and control input
    print(sprintf("Time: %.2f, Position: %.2f, Speed: %.2f, Control: %.2f", t, state["position"], state["velocity"], control_inputs[i]))
  }

  return(results)
}

save_results <- function(results, stability_metrics, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(results, file.path(output_dir, "controlled_smd_simulation_R.csv"), row.names = FALSE)

  # Flatten stability metrics
  flattened_metrics <- do.call(rbind, lapply(names(stability_metrics), function(name) {
    value <- stability_metrics[[name]]
    if (is.vector(value) && length(value) > 1) {
      return(data.frame(Criterion = paste0(name, " [", seq_along(value), "]"), Value = as.character(value)))
    } else {
      return(data.frame(Criterion = name, Value = as.character(value)))
    }
  }))

  write.csv(flattened_metrics, file.path(output_dir, "controlled_smd_stability_metrics_R.csv"), row.names = FALSE)

  print("\nClosed-Loop Stability Analysis:")
  print(flattened_metrics)
}

# Main execution
main <- function() {
  params <- list(m = 100, k = 50, d = 50, 
                 kp = 200, ki = 50, kd = 100)

  sim_params <- list(start_time = 0, end_time = 20, step_size = 0.01, 
                     input_start = 2, input_end = 2.5, 
                     input_start_val = 0, input_end_val = 1)

  stability_metrics <- analyze_closed_loop_stability(params)

  results <- simulate_with_control(params, sim_params)

  save_results(results, stability_metrics, "../outputs/data")

  print(head(results))
}

main()


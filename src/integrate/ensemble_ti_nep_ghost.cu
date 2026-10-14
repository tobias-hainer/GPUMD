/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "ensemble_ti_nep_ghost.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/run_input.cuh"
#include <cstdio>
#include <memory>

namespace
{

// gather the positions of the non-ghost atoms into the compact arrays
static __global__ void gpu_gather_positions(
  int N_keep,
  const int* keep_indices,
  const double* x_full,
  const double* y_full,
  const double* z_full,
  double* x_compact,
  double* y_compact,
  double* z_compact)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N_keep) {
    const int i = keep_indices[j];
    x_compact[j] = x_full[i];
    y_compact[j] = y_full[i];
    z_compact[j] = z_full[i];
  }
}

// scatter the results of the compact system back to the full arrays
// (the ghost atom slots stay zero)
static __global__ void gpu_scatter_nep2_results(
  int N_keep,
  const int* keep_indices,
  const double* pot_compact,
  const double* fx_compact,
  const double* fy_compact,
  const double* fz_compact,
  double* pot_full,
  double* fx_full,
  double* fy_full,
  double* fz_full)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N_keep) {
    const int i = keep_indices[j];
    pot_full[i] = pot_compact[j];
    fx_full[i] = fx_compact[j];
    fy_full[i] = fy_compact[j];
    fz_full[i] = fz_compact[j];
  }
}

// F = (1 - lambda) * F1 + lambda * F2
// for the ghost atoms, F2 is the force of an Einstein spring
static __global__ void gpu_mix_forces(
  int number_of_atoms,
  Box box,
  double lambda,
  const bool* is_ghost,
  double* espring,
  double* k,
  double* x,
  double* y,
  double* z,
  double* x0,
  double* y0,
  double* z0,
  double* fx,
  double* fy,
  double* fz,
  double* fx_nep2,
  double* fy_nep2,
  double* fz_nep2)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    espring[i] = 0.0;
    const double lm = lambda;
    const double gm = 1.0 - lambda;
    if (!is_ghost[i]) {
      fx[i] = gm * fx[i] + lm * fx_nep2[i];
      fy[i] = gm * fy[i] + lm * fy_nep2[i];
      fz[i] = gm * fz[i] + lm * fz_nep2[i];
    } else {
      double dx = x[i] - x0[i];
      double dy = y[i] - y0[i];
      double dz = z[i] - z0[i];
      apply_mic(box, dx, dy, dz);
      fx[i] = gm * fx[i] + lm * (-k[i] * dx);
      fy[i] = gm * fy[i] + lm * (-k[i] * dy);
      fz[i] = gm * fz[i] + lm * (-k[i] * dz);
      espring[i] = 0.5 * k[i] * (dx * dx + dy * dy + dz * dz);
    }
  }
}

static __global__ void gpu_get_espring_sum_nep(const int N, double* espring)
{
  // <<<1, 1024>>>
  int tid = threadIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; patch++) {
    int n = tid + patch * 1024;
    if (n < N)
      s_data[tid] += espring[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    espring[0] = s_data[0];
}

static __global__ void
initialize_properties(int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
  }
}

// the potentials accumulate onto their outputs, so zero them before each evaluation
static __global__ void initialize_properties_compact(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    for (int d = 0; d < 9; ++d) {
      g_virial[n1 + d * N] = 0.0;
    }
  }
}

// Beta smoothstep S_n(x) = I_x(n + 1, n + 1) for n = 4..8 (n = 4 is the smootherstep)
// S_n(x) = 1 - S_n(1 - x) is used for x > 1/2 to avoid round-off near x = 1
static double smoothstep(const int order, double x)
{
  if (x < 0.0)
    x = 0.0;
  if (x > 1.0)
    x = 1.0;
  if (x > 0.5)
    return 1.0 - smoothstep(order, 1.0 - x);
  const double x2 = x * x;
  const double x4 = x2 * x2;
  switch (order) {
    case 4: // 126x^5 - 420x^6 + 540x^7 - 315x^8 + 70x^9
      return x4 * x * (126.0 + x * (-420.0 + x * (540.0 + x * (-315.0 + x * 70.0))));
    case 5: // 462x^6 - 1980x^7 + 3465x^8 - 3080x^9 + 1386x^10 - 252x^11
      return x4 * x2 *
             (462.0 + x * (-1980.0 + x * (3465.0 + x * (-3080.0 + x * (1386.0 + x * -252.0)))));
    case 6: // 1716x^7 - 9009x^8 + 20020x^9 - 24024x^10 + 16380x^11 - 6006x^12 + 924x^13
      return x4 * x2 * x *
             (1716.0 +
              x * (-9009.0 +
                   x * (20020.0 + x * (-24024.0 + x * (16380.0 + x * (-6006.0 + x * 924.0))))));
    case 7: // 6435x^8 - 40040x^9 + 108108x^10 - 163800x^11 + 150150x^12 - 83160x^13
            // + 25740x^14 - 3432x^15
      return x4 * x4 *
             (6435.0 +
              x * (-40040.0 +
                   x * (108108.0 +
                        x * (-163800.0 +
                             x * (150150.0 + x * (-83160.0 + x * (25740.0 + x * -3432.0)))))));
    case 8: // 24310x^9 - 175032x^10 + 556920x^11 - 1021020x^12 + 1178100x^13
            // - 875160x^14 + 408408x^15 - 109395x^16 + 12870x^17
      return x4 * x4 * x *
             (24310.0 +
              x *
                (-175032.0 +
                 x *
                   (556920.0 +
                    x * (-1021020.0 +
                         x * (1178100.0 +
                              x * (-875160.0 + x * (408408.0 + x * (-109395.0 + x * 12870.0))))))));
    default:
      PRINT_INPUT_ERROR("path_order must be an integer between 4 and 8.");
      return x; // unreachable
  }
}

// S_n'(x) = C_n * x^n * (1 - x)^n with C_n = (2n + 1)! / (n!)^2
static double dsmoothstep(const int order, double x)
{
  if (x < 0.0)
    x = 0.0;
  if (x > 1.0)
    x = 1.0;
  const double u = x * (1.0 - x);
  const double u2 = u * u;
  const double u4 = u2 * u2;
  switch (order) {
    case 4:
      return 630.0 * u4;
    case 5:
      return 2772.0 * u4 * u;
    case 6:
      return 12012.0 * u4 * u2;
    case 7:
      return 51480.0 * u4 * u2 * u;
    case 8:
      return 218790.0 * u4 * u4;
    default:
      PRINT_INPUT_ERROR("path_order must be an integer between 4 and 8.");
      return 0.0; // unreachable
  }
}

} // namespace

// ensemble ti_nep_ghost neq [keywords]
// ensemble ti_nep_ghost eq <N_lambda> <equil_steps> <prod_steps> <forward|backward> [keywords]
// for old inputs, "ti_nep_ghost [keywords]" is the neq mode and "ti_eq_nep_ghost ..." the eq mode
Ensemble_TI_Nep_Ghost::Ensemble_TI_Nep_Ghost(const std::vector<std::string>& tokens)
{
  temperature_coupling = 100;
  if (tokens[1] == "ti_eq_nep_ghost") {
    parse_eq(tokens, 2);
  } else if (tokens.size() > 2 && tokens[2] == "neq") {
    parse_neq(tokens, 3);
  } else if (tokens.size() > 2 && tokens[2] == "eq") {
    parse_eq(tokens, 3);
  } else {
    printf("No mode given for ensemble ti_nep_ghost, using the neq mode.\n");
    parse_neq(tokens, 2);
  }

  type = EnsembleType::NVT_LAN;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

// keywords valid in both modes, returns false if tokens[i] is not one of them
bool Ensemble_TI_Nep_Ghost::parse_common_keyword(const std::vector<std::string>& tokens, int i)
{
  if (tokens[i] == "temp") {
    if (!is_valid_real(tokens[i + 1], &temperature))
      PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
  } else if (tokens[i] == "tperiod") {
    if (!is_valid_real(tokens[i + 1], &temperature_coupling))
      PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
  } else if (tokens[i] == "ghost_spring") {
    if (!is_valid_real(tokens[i + 1], &ghost_k))
      PRINT_INPUT_ERROR("Wrong inputs for ghost_spring keyword.");
  } else if (tokens[i] == "path_order") {
    if (!is_valid_int(tokens[i + 1], &path_order))
      PRINT_INPUT_ERROR("Wrong inputs for path_order keyword.");
    if (path_order < 4 || path_order > 8)
      PRINT_INPUT_ERROR("path_order must be an integer between 4 and 8.");
  } else {
    return false;
  }
  return true;
}

void Ensemble_TI_Nep_Ghost::parse_neq(const std::vector<std::string>& tokens, int i)
{
  mode = Mode::NEQ;
  const int num_params = tokens.size();
  while (i < num_params) {
    if (i + 1 >= num_params) {
      PRINT_INPUT_ERROR("Missing value for the last keyword of ensemble ti_nep_ghost.");
    }
    if (parse_common_keyword(tokens, i)) {
    } else if (tokens[i] == "tswitch") {
      auto_switch = false;
      if (!is_valid_int(tokens[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
    } else if (tokens[i] == "tequil") {
      auto_switch = false;
      if (!is_valid_int(tokens[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
    } else {
      PRINT_INPUT_ERROR("Unknown keyword for the neq mode of ensemble ti_nep_ghost.");
    }
    i += 2;
  }
  if ((t_switch < 0 && t_equil >= 0) || (t_switch >= 0 && t_equil < 0)) {
    PRINT_INPUT_ERROR(
      "Error: Please specify either both t_switch and t_equil, or neither (to let the program "
      "auto-determine)");
  }
  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  printf(
    "Nonequilibrium (neq) mode. Coupling is linear in lambda; path_order = %d is the order of "
    "the time-switching schedule lambda(t) (Beta smoothstep; 4 = smootherstep).\n",
    path_order);
}

void Ensemble_TI_Nep_Ghost::parse_eq(const std::vector<std::string>& tokens, int i)
{
  mode = Mode::EQ;
  const int num_params = tokens.size();
  if (num_params < i + 4) {
    PRINT_INPUT_ERROR(
      "The eq mode of ensemble ti_nep_ghost requires N_lambda, equil_steps, prod_steps, and "
      "direction.");
  }

  if (!is_valid_int(tokens[i], &N_lambda)) {
    PRINT_INPUT_ERROR("Wrong input for N_lambda.");
  }
  if (N_lambda < 2) {
    PRINT_INPUT_ERROR("N_lambda should be >= 2.");
  }

  if (!is_valid_int(tokens[i + 1], &equil_steps)) {
    PRINT_INPUT_ERROR("Wrong input for equil_steps.");
  }
  if (equil_steps < 0) {
    PRINT_INPUT_ERROR("equil_steps should be >= 0.");
  }

  if (!is_valid_int(tokens[i + 2], &prod_steps)) {
    PRINT_INPUT_ERROR("Wrong input for prod_steps.");
  }
  if (prod_steps <= 0) {
    PRINT_INPUT_ERROR("prod_steps should be > 0.");
  }

  if (tokens[i + 3] == "forward") {
    reverse_direction = false;
  } else if (tokens[i + 3] == "backward") {
    reverse_direction = true;
  } else {
    PRINT_INPUT_ERROR("direction should be 'forward' or 'backward'.");
  }

  i += 4;
  while (i < num_params) {
    if (i + 1 >= num_params) {
      PRINT_INPUT_ERROR("Missing value for the last keyword of ensemble ti_nep_ghost.");
    }
    if (parse_common_keyword(tokens, i)) {
    } else if (tokens[i] == "pre_equil") {
      if (!is_valid_int(tokens[i + 1], &pre_equil_steps))
        PRINT_INPUT_ERROR("Wrong inputs for pre_equil keyword.");
      if (pre_equil_steps < 0)
        PRINT_INPUT_ERROR("pre_equil should be >= 0.");
    } else {
      PRINT_INPUT_ERROR("Unknown keyword for the eq mode of ensemble ti_nep_ghost.");
    }
    i += 2;
  }

  steps_per_window = equil_steps + prod_steps;

  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  printf(
    "Equilibrium (eq) mode: %d lambda windows, %d equilibration + %d production steps per "
    "window, direction: %s.\n",
    N_lambda,
    equil_steps,
    prod_steps,
    reverse_direction ? "backward (lambda: 1 -> 0)" : "forward (lambda: 0 -> 1)");
  printf(
    "Coupling is linear in lambda; lambda windows placed at S_n(k / (N_lambda - 1)), Beta "
    "smoothstep of order n = path_order = %d.\n",
    path_order);
  if (pre_equil_steps > 0) {
    if (reverse_direction) {
      printf(
        "Pre-equilibration: %d steps ramping lambda 0 -> 1 (smootherstep) before the ladder "
        "starts.\n",
        pre_equil_steps);
    } else {
      printf(
        "Pre-equilibration: %d steps at lambda = 0 before the ladder starts.\n", pre_equil_steps);
    }
  }
}

void Ensemble_TI_Nep_Ghost::init_neq(const int number_of_steps)
{
  if (auto_switch) {
    t_switch = (int)(number_of_steps * 0.4);
    t_equil = (int)(number_of_steps * 0.1);
  } else
    printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Nonequilibrium thermodynamic integration (ghost-atom / vacancy path): t_switch is %d "
    "timestep, t_equil is %d timesteps.\n",
    t_switch,
    t_equil);
}

void Ensemble_TI_Nep_Ghost::init_eq(const int number_of_steps)
{
  if (initialized) {
    PRINT_INPUT_ERROR(
      "ti_nep_ghost eq: init() called a second time on the same ensemble instance -- issue a "
      "new 'ensemble ti_nep_ghost' line before each 'run' that should start a fresh ladder.");
  }
  initialized = true;

  const int expected_total_steps = pre_equil_steps + N_lambda * steps_per_window;
  if (number_of_steps != expected_total_steps) {
    char message[256];
    snprintf(
      message,
      sizeof(message),
      "ti_nep_ghost eq requires run's step count to equal pre_equil_steps + N_lambda * "
      "(equil_steps + prod_steps) = %d + %d * %d = %d, but run specified %d steps.",
      pre_equil_steps,
      N_lambda,
      steps_per_window,
      expected_total_steps,
      number_of_steps);
    PRINT_INPUT_ERROR(message);
  }

  // windows are uniform in u, with lambda = S_n(u)
  u_grid.resize(N_lambda);
  lambda_grid.resize(N_lambda);
  for (int k = 0; k < N_lambda; ++k) {
    u_grid[k] = (double)k / (double)(N_lambda - 1);
    lambda_grid[k] = smoothstep(path_order, u_grid[k]);
  }
  lambda_grid[0] = 0.0;
  lambda_grid[N_lambda - 1] = 1.0;
  window_sum.assign(N_lambda, 0.0);
  window_sumsq.assign(N_lambda, 0.0);
  window_n.assign(N_lambda, 0);
  window_mean.assign(N_lambda, 0.0);

  // check that the lambda grid is strictly increasing
  double dlambda_min = 1.0;
  double dlambda_max = 0.0;
  for (int k = 0; k < N_lambda - 1; ++k) {
    const double d = lambda_grid[k + 1] - lambda_grid[k];
    if (!(d > 0.0)) {
      char message[256];
      snprintf(
        message,
        sizeof(message),
        "ti_nep_ghost eq: lambda grid is not strictly increasing between windows %d and %d "
        "(lambda = %.17g, %.17g); reduce N_lambda or path_order = %d.",
        k,
        k + 1,
        lambda_grid[k],
        lambda_grid[k + 1],
        path_order);
      PRINT_INPUT_ERROR(message);
    }
    if (d < dlambda_min)
      dlambda_min = d;
    if (d > dlambda_max)
      dlambda_max = d;
  }
  printf("Lambda windows (path_order = %d):\n", path_order);
  printf("    %6s %14s %22s\n", "window", "u", "lambda");
  for (int k = 0; k < N_lambda; ++k) {
    printf("    %6d %14.10f %22.16e\n", k, u_grid[k], lambda_grid[k]);
  }
  printf("Smallest Delta lambda: %e; largest Delta lambda: %e.\n", dlambda_min, dlambda_max);
}

void Ensemble_TI_Nep_Ghost::init_ghost(
  const std::vector<Group>& group, const Atom& atom, const GPU_Vector<double>& thermo)
{
  const int N = atom.number_of_atoms;
  N_full = N;

  // the ghost atoms are group 1 of grouping method 0
  const int grouping_method = 0;
  const int ghost_group_id = 1;
  if (group.size() <= grouping_method) {
    PRINT_INPUT_ERROR(
      "ti_nep_ghost requires a grouping method marking the ghost atom(s) (group 1).");
  }
  if (group[grouping_method].number <= ghost_group_id) {
    PRINT_INPUT_ERROR(
      "ti_nep_ghost requires at least 2 groups in grouping method 0 (0 = keep, 1 = ghost).");
  }
  const int ghost_size = group[grouping_method].cpu_size[ghost_group_id];
  const int ghost_offset = group[grouping_method].cpu_size_sum[ghost_group_id];
  if (ghost_size < 1) {
    PRINT_INPUT_ERROR("ti_nep_ghost requires at least one atom in the ghost group.");
  }
  N_keep = N - ghost_size;

  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo.size());

  ghost_ids.resize(ghost_size);
  std::vector<int> ghost_flag(N, 0);
  for (int n = 0; n < ghost_size; ++n) {
    ghost_ids[n] = group[grouping_method].cpu_contents[ghost_offset + n];
    ghost_flag[ghost_ids[n]] = 1;
  }

  std::vector<int> cpu_keep_indices(N_keep);
  {
    int j = 0;
    for (int n = 0; n < N; ++n) {
      if (!ghost_flag[n]) {
        cpu_keep_indices[j++] = n;
      }
    }
  }
  keep_indices.resize(N_keep);
  keep_indices.copy_from_host(cpu_keep_indices.data());

  std::unique_ptr<bool[]> cpu_is_ghost(new bool[N]);
  for (int n = 0; n < N; ++n) {
    cpu_is_ghost[n] = ghost_flag[n] != 0;
  }
  is_ghost.resize(N);
  is_ghost.copy_from_host(cpu_is_ghost.get());

  // spring constants, nonzero only for the ghost atoms
  std::vector<double> cpu_k(N, 0.0);
  for (const int atom_id : ghost_ids) {
    cpu_k[atom_id] = ghost_k;
  }
  gpu_k.resize(N);
  gpu_k.copy_from_host(cpu_k.data());

  gpu_espring.resize(N, 0.0);

  position_0.resize(3 * N);
  CHECK(gpuMemcpy(
    position_0.data(),
    atom.position_per_atom.data(),
    sizeof(double) * position_0.size(),
    gpuMemcpyDeviceToDevice));

  // compact arrays for the system without the ghost atoms
  std::vector<int> cpu_type_compact(N_keep);
  for (int j = 0; j < N_keep; ++j) {
    cpu_type_compact[j] = atom.cpu_type[cpu_keep_indices[j]];
  }
  type_compact.resize(N_keep);
  type_compact.copy_from_host(cpu_type_compact.data());

  RunInput run_input("run.in");
  const std::string nep_file = get_first_potential_filename(run_input);
  nep_ghost = std::make_unique<NEP>(nep_file.c_str(), N_keep, run_input);
  if (nep_ghost->nep_model_type != 0) {
    PRINT_INPUT_ERROR("ti_nep_ghost requires a regular NEP model.");
  }
  nep_ghost->N1 = 0;
  nep_ghost->N2 = N_keep;

  position_compact.resize(3 * N_keep);
  potential_compact.resize(N_keep);
  force_compact.resize(3 * N_keep);
  virial_compact.resize(9 * N_keep);

  potential_nep2.resize(N);
  force_nep2.resize(3 * N);

  printf(
    "Ghost-atom TI: %d atom(s) will be switched off along the path (vacancy formation), "
    "spring constant = %f eV/A^2.\n",
    ghost_size,
    ghost_k);
}

void Ensemble_TI_Nep_Ghost::initialize_before_first_step(
  const double,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box&,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (mode == Mode::NEQ) {
    init_neq(number_of_steps);
  } else {
    init_eq(number_of_steps);
  }

  init_ghost(group, atom, thermo);

  if (mode == Mode::NEQ) {
    output_file = my_fopen("ti_nep_ghost.csv", "w");
    fprintf(output_file, "lambda,dlambda,pe,pe_nep2,espring\n");
  } else {
    dudl_file = my_fopen("ti_eq_nep_ghost_dudl.csv", "w");
    fprintf(dudl_file, "global_step,window_index,u,lambda,dU_dlambda\n");
    summary_file = my_fopen("ti_eq_nep_ghost_summary.csv", "w");
    fprintf(
      summary_file,
      "window_index,u,lambda,mean_dU_dlambda,standard_error,n_samples,running_integral\n");
  }
}

void Ensemble_TI_Nep_Ghost::find_thermo(
  const Box& box, const std::vector<Group>& group, const Atom& atom, GPU_Vector<double>& thermo)
{
  Ensemble::find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
  thermo.copy_to_host(thermo_cpu.data());
  pe = thermo_cpu[1];
}

void Ensemble_TI_Nep_Ghost::find_thermo_nep2(const int number_of_atoms)
{
  // the ghost atom slots are zero
  std::vector<double> host_potential(number_of_atoms);
  potential_nep2.copy_to_host(host_potential.data());
  double sum = 0.0;
  for (int n = 0; n < number_of_atoms; ++n) {
    sum += host_potential[n];
  }
  pe_nep2 = sum;
}

Ensemble_TI_Nep_Ghost::~Ensemble_TI_Nep_Ghost(void) { close_output_files(false); }

void Ensemble_TI_Nep_Ghost::finalize_run(const Atom& atom, const Box&)
{
  double kT = K_B * temperature;

  // free energy of the Einstein oscillators replacing the ghost atoms
  for (const int atom_id : ghost_ids) {
    double w = pow(ghost_k / atom.cpu_mass[atom_id], 0.5);
    E_Ein += log(w * HBAR / kT);
  }
  E_Ein = 3 * kT * E_Ein;

  double F_diff = E_diff - E_Ein;

  FILE* yaml_file =
    my_fopen(mode == Mode::NEQ ? "ti_nep_ghost.yaml" : "ti_eq_nep_ghost.yaml", "w");
  fprintf(yaml_file, "mode: %s\n", mode == Mode::NEQ ? "neq" : "eq");
  if (mode == Mode::EQ) {
    fprintf(yaml_file, "direction: %s\n", reverse_direction ? "backward" : "forward");
    fprintf(yaml_file, "N_lambda: %d\n", N_lambda);
  }
  fprintf(yaml_file, "path_order: %d\n", path_order);
  if (mode == Mode::EQ) {
    fprintf(yaml_file, "windows_completed: %d\n", windows_completed);
  }
  fprintf(yaml_file, "E_diff: %f\n", E_diff);
  fprintf(yaml_file, "E_Ein: %f\n", E_Ein);
  fprintf(yaml_file, "F_diff: %f\n", F_diff);
  fprintf(yaml_file, "T: %f\n", temperature);

  close_output_files(true);
  fclose(yaml_file);

  printf("\n");
  printf("-----------------------------------------------------------------------\n");
  if (mode == Mode::EQ) {
    printf("Equilibrium thermodynamic integration (ghost-atom / vacancy path)\n");
    printf(
      "Direction: %s\n",
      reverse_direction ? "backward (lambda: 1 -> 0)" : "forward (lambda: 0 -> 1)");
    if (windows_completed < N_lambda) {
      printf(
        "Warning: run ended after %d of %d windows; the integral below is partial (covers only "
        "the completed lambda range).\n",
        windows_completed,
        N_lambda);
    }
  } else {
    printf("Nonequilibrium thermodynamic integration (ghost-atom / vacancy path)\n");
  }
  printf("TI integral (raw):                     %f eV\n", E_diff);
  printf("Einstein spring reference (E_Ein):     %f eV\n", E_Ein);
  printf("Difference in Helmholtz free energy:   %f eV\n", F_diff);
  if (mode == Mode::NEQ) {
    printf("These values are stored in ti_nep_ghost.yaml.\n");
  } else {
    printf(
      "These values are stored in ti_eq_nep_ghost.yaml; per-step and per-window data are in\n");
    printf("ti_eq_nep_ghost_dudl.csv and ti_eq_nep_ghost_summary.csv.\n");
  }
  printf("-----------------------------------------------------------------------\n");
}

void Ensemble_TI_Nep_Ghost::close_output_files(const bool print_message)
{
  if (print_message) {
    printf("Closing ti_nep_ghost output files...\n");
  }
  for (FILE** file : {&output_file, &dudl_file, &summary_file}) {
    if (*file != nullptr) {
      fclose(*file);
      *file = nullptr;
    }
  }
}

void Ensemble_TI_Nep_Ghost::add_mixed_force(const Box& box, Atom& atom)
{
  const int N = atom.number_of_atoms;
  gpu_mix_forces<<<(N - 1) / 128 + 1, 128>>>(
    N,
    box,
    lambda,
    is_ghost.data(),
    gpu_espring.data(),
    gpu_k.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N,
    atom.force_per_atom.data(),
    atom.force_per_atom.data() + N,
    atom.force_per_atom.data() + 2 * N,
    force_nep2.data(),
    force_nep2.data() + N,
    force_nep2.data() + 2 * N);
  GPU_CHECK_KERNEL
}

double Ensemble_TI_Nep_Ghost::get_espring_sum(const int number_of_atoms)
{
  double temp;
  gpu_get_espring_sum_nep<<<1, 1024>>>(number_of_atoms, gpu_espring.data());
  GPU_CHECK_KERNEL
  gpu_espring.copy_to_host(&temp, 1);
  return temp;
}

// neq: lambda(t) for the step
void Ensemble_TI_Nep_Ghost::find_lambda(const int step)
{
  const int t = step - t_equil;
  const double r_switch = 1.0 / t_switch;

  if (t < 0) {
    // initial equilibration at lambda = 0
    lambda = 0.0;
    dlambda = 0.0;
  } else if (t <= t_switch) {
    // forward switch 0 -> 1
    lambda = smoothstep(path_order, t * r_switch);
    dlambda = dsmoothstep(path_order, t * r_switch) / t_switch;
  } else if (t < t_equil + t_switch) {
    // plateau at lambda = 1
    lambda = 1.0;
    dlambda = 0.0;
  } else if (t <= t_equil + 2 * t_switch) {
    // backward switch 1 -> 0
    const double x = 1.0 - (t - t_switch - t_equil) * r_switch;
    lambda = smoothstep(path_order, x);
    dlambda = -dsmoothstep(path_order, x) / t_switch;
  } else {
    // after the backward leg, back at lambda = 0
    lambda = 0.0;
    dlambda = 0.0;
  }
}

// neq: add the work of this step to the integral
void Ensemble_TI_Nep_Ghost::accumulate_neq(const int number_of_atoms)
{
  if (dlambda == 0.0) {
    return;
  }
  const int N = number_of_atoms;
  espring = get_espring_sum(N);

  // U = (1 - lambda) * U1 + lambda * U2, so dU/dlambda = U2 - U1
  const double dudl = (pe_nep2 + espring) - pe;

  fprintf(output_file, "%e,%e,%e,%e,%e\n", lambda, dlambda, pe / N, pe_nep2 / N, espring / N);

  E_diff += 0.5 * dudl * fabs(dlambda);
}

// eq: window and lambda for the step
void Ensemble_TI_Nep_Ghost::update_schedule(const int step)
{
  if (step < pre_equil_steps) {
    // move to the starting lambda of the ladder
    in_pre_equil = true;
    window = reverse_direction ? (N_lambda - 1) : 0;
    canonical_index = window;
    step_in_window = -1;
    if (reverse_direction) {
      // ramp smoothly from lambda = 0 to 1 to avoid a sudden change of the forces
      const double frac = (double)(step) / (double)pre_equil_steps;
      lambda = smoothstep(4, frac);
    } else {
      lambda = 0.0;
    }
    in_production = false;
    return;
  }
  in_pre_equil = false;
  const int ladder_step = step - pre_equil_steps;
  window = ladder_step / steps_per_window;
  step_in_window = ladder_step % steps_per_window;
  canonical_index = reverse_direction ? (N_lambda - 1 - window) : window;
  lambda = lambda_grid[canonical_index];
  in_production = step_in_window >= equil_steps;
}

// eq: sample dU/dlambda in the production part of a window
void Ensemble_TI_Nep_Ghost::accumulate_eq(const int step, const int number_of_atoms)
{
  if (!in_production) {
    return;
  }
  espring = get_espring_sum(number_of_atoms);

  // U = (1 - lambda) * U1 + lambda * U2, so dU/dlambda = U2 - U1
  const double dudl = (pe_nep2 + espring) - pe;

  window_sum[canonical_index] += dudl;
  window_sumsq[canonical_index] += dudl * dudl;
  window_n[canonical_index] += 1;

  fprintf(
    dudl_file, "%d,%d,%e,%e,%e\n", step, canonical_index, u_grid[canonical_index], lambda, dudl);
  fflush(dudl_file);

  if (step_in_window == steps_per_window - 1) {
    close_window();
  }
}

void Ensemble_TI_Nep_Ghost::close_window()
{
  const int n = window_n[canonical_index];
  const double mean = window_sum[canonical_index] / n;
  double standard_error = 0.0;
  if (n > 1) {
    double variance = (window_sumsq[canonical_index] - n * mean * mean) / (n - 1);
    if (variance < 0.0) {
      variance = 0.0; // round-off
    }
    standard_error = sqrt(variance / n);
  }
  window_mean[canonical_index] = mean;
  windows_completed += 1;

  // trapezoidal rule over the completed windows
  int lo, hi;
  if (reverse_direction) {
    lo = N_lambda - windows_completed;
    hi = N_lambda - 1;
  } else {
    lo = 0;
    hi = windows_completed - 1;
  }
  double running_integral = 0.0;
  for (int k = lo; k < hi; ++k) {
    running_integral +=
      0.5 * (window_mean[k] + window_mean[k + 1]) * (lambda_grid[k + 1] - lambda_grid[k]);
  }
  E_diff = running_integral;

  fprintf(
    summary_file,
    "%d,%e,%e,%e,%e,%d,%e\n",
    canonical_index,
    u_grid[canonical_index],
    lambda_grid[canonical_index],
    mean,
    standard_error,
    n,
    running_integral);
  fflush(summary_file);
}

// evaluate the NEP potential on the system without the ghost atoms
void Ensemble_TI_Nep_Ghost::compute_ghost_free_system(Box& box, Atom& atoms)
{
  const int N = atoms.number_of_atoms;

  initialize_properties<<<(N - 1) / 128 + 1, 128>>>(
    N, force_nep2.data(), force_nep2.data() + N, force_nep2.data() + 2 * N, potential_nep2.data());
  GPU_CHECK_KERNEL

  gpu_gather_positions<<<(N_keep - 1) / 128 + 1, 128>>>(
    N_keep,
    keep_indices.data(),
    atoms.position_per_atom.data(),
    atoms.position_per_atom.data() + N,
    atoms.position_per_atom.data() + 2 * N,
    position_compact.data(),
    position_compact.data() + N_keep,
    position_compact.data() + 2 * N_keep);
  GPU_CHECK_KERNEL

  initialize_properties_compact<<<(N_keep - 1) / 128 + 1, 128>>>(
    N_keep,
    force_compact.data(),
    force_compact.data() + N_keep,
    force_compact.data() + 2 * N_keep,
    potential_compact.data(),
    virial_compact.data());
  GPU_CHECK_KERNEL

  nep_ghost->compute(
    box, type_compact, position_compact, potential_compact, force_compact, virial_compact);

  gpu_scatter_nep2_results<<<(N_keep - 1) / 128 + 1, 128>>>(
    N_keep,
    keep_indices.data(),
    potential_compact.data(),
    force_compact.data(),
    force_compact.data() + N_keep,
    force_compact.data() + 2 * N_keep,
    potential_nep2.data(),
    force_nep2.data(),
    force_nep2.data() + N,
    force_nep2.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void Ensemble_TI_Nep_Ghost::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  const int N = atoms.number_of_atoms;

  compute_ghost_free_system(box, atoms);

  if (mode == Mode::NEQ) {
    find_lambda(step);
  } else {
    update_schedule(step);
  }
  find_thermo(box, group, atoms, thermo);
  find_thermo_nep2(N);

  add_mixed_force(box, atoms);

  if (mode == Mode::NEQ) {
    accumulate_neq(N);
  } else {
    accumulate_eq(step, N);
  }

  Ensemble_LAN::compute2(time_step, step, number_of_steps, group, box, atoms, thermo, force);
}

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

#pragma once
#include "ensemble_lan.cuh"
#include "force/force.cuh"
#include "force/nep.cuh"
#include "langevin_utilities.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <math.h>
#include <memory>
#include <string>
#include <vector>

// thermodynamic integration for vacancy formation: the ghost atoms (group 1 of grouping
// method 0) are switched from the NEP potential to Einstein springs
// U = (1 - lambda) * U1 + lambda * U2
// U1: the NEP potential on the full system
// U2: the NEP potential on the system without the ghost atoms, plus the springs
// two modes:
// neq: lambda(t) is a Beta smoothstep of order path_order in time (nonequilibrium switching)
// eq:  lambda is held fixed in N_lambda windows at lambda_k = S_n(k / (N_lambda - 1)); each
//      window runs equil_steps, then samples dU/dlambda for prod_steps; the window means are
//      integrated with the trapezoidal rule
class Ensemble_TI_Nep_Ghost : public Ensemble_LAN
{
public:
  Ensemble_TI_Nep_Ghost(const std::vector<std::string>& tokens);
  ~Ensemble_TI_Nep_Ghost(void) override;

  void initialize_before_first_step(
    const double time_step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo,
    Force& force) override;

  void finalize_run(const Atom& atom, const Box& box) override;

protected:
  enum class Mode { NEQ, EQ };
  Mode mode = Mode::NEQ;

  // parameters shared by both modes
  // order of the smoothstep, 4 to 8
  int path_order = 4;
  // spring constant of the ghost atoms
  double ghost_k = 10.0;

  // neq mode
  FILE* output_file = nullptr;
  int t_equil = -1, t_switch = -1;
  bool auto_switch = true;
  double dlambda = 0;

  // eq mode
  FILE* dudl_file = nullptr;
  FILE* summary_file = nullptr;
  int N_lambda = 0;
  int equil_steps = 0;
  int prod_steps = 0;
  bool reverse_direction = false; // false: lambda 0->1, true: lambda 1->0
  int steps_per_window = 0;
  bool initialized = false;
  int pre_equil_steps = 0;  // steps at the starting lambda before the first window, not sampled
  int window = -1;          // index in execution order
  int canonical_index = -1; // index on the ascending lambda grid
  int step_in_window = -1;
  bool in_production = false;
  bool in_pre_equil = false;
  int windows_completed = 0;
  // indexed by canonical_index
  std::vector<double> u_grid;      // k / (N_lambda - 1)
  std::vector<double> lambda_grid; // S_n(u)
  std::vector<double> window_sum;
  std::vector<double> window_sumsq;
  std::vector<int> window_n;
  std::vector<double> window_mean;

  double lambda = 0;
  double pe = 0, pe_nep2 = 0, espring = 0;
  double E_diff = 0, E_Ein = 0;

  int N_full = 0;
  int N_keep = 0;
  std::vector<int> ghost_ids;

  // size N_full
  GPU_Vector<bool> is_ghost;
  GPU_Vector<double> gpu_k;
  GPU_Vector<double> gpu_espring;
  GPU_Vector<double> position_0;

  // compact system without the ghost atoms, size N_keep
  std::unique_ptr<NEP> nep_ghost;
  GPU_Vector<int> keep_indices; // compact index -> full index
  GPU_Vector<int> type_compact;
  GPU_Vector<double> position_compact;
  GPU_Vector<double> potential_compact;
  GPU_Vector<double> force_compact;
  GPU_Vector<double> virial_compact;

  // results of the second potential, size N_full
  GPU_Vector<double> potential_nep2;
  GPU_Vector<double> force_nep2;

  std::vector<double> thermo_cpu;

  void parse_neq(const std::vector<std::string>& tokens, int i);
  void parse_eq(const std::vector<std::string>& tokens, int i);
  bool parse_common_keyword(const std::vector<std::string>& tokens, int i);

  void init_neq(const int number_of_steps);
  void init_eq(const int number_of_steps);
  void init_ghost(
    const std::vector<Group>& group, const Atom& atom, const GPU_Vector<double>& thermo);

  void compute_ghost_free_system(Box& box, Atom& atoms);
  void find_thermo(
    const Box& box, const std::vector<Group>& group, const Atom& atom, GPU_Vector<double>& thermo);
  void find_thermo_nep2(const int number_of_atoms);
  void add_mixed_force(const Box& box, Atom& atom);
  double get_espring_sum(const int number_of_atoms);

  void find_lambda(const int step);
  void accumulate_neq(const int number_of_atoms);
  void update_schedule(const int step);
  void accumulate_eq(const int step, const int number_of_atoms);
  void close_window();

private:
  void close_output_files(const bool print_message);
};

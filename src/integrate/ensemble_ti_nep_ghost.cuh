/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
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

// nonequilibrium TI for vacancy formation: the ghost atoms (group 1 of grouping
// method 0) are switched from the NEP potential to Einstein springs
// U = (1 - lambda) * U1 + lambda * U2, with lambda(t) a Beta smoothstep of order path_order
// U1: the first potential on the full system
// U2: the second potential on the system without the ghost atoms, plus the springs
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

  void find_thermo(
    const Box& box, const std::vector<Group>& group, const Atom& atom, GPU_Vector<double>& thermo);
  void find_thermo_nep2(const int number_of_atoms);
  void add_mixed_force(const Box& box, Atom& atom);
  double get_espring_sum(const int number_of_atoms);
  void init(
    const int number_of_steps,
    const std::vector<Group>& group,
    const Atom& atom,
    const GPU_Vector<double>& thermo);
  void find_lambda(const int step);
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  FILE* output_file = nullptr;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double pe = 0, pe_nep2 = 0, espring = 0;
  double E_diff = 0, E_Ein = 0;
  bool auto_switch = true;

  // order of the switching function, 4 to 8
  int path_order = 4;

  // spring constant of the ghost atoms
  double ghost_k = 10.0;

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

private:
  void close_output_file(const bool print_message);
};
